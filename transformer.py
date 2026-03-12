import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass

def generate_rope(max_seq_len, head_dim):
  base = 10000
  theta = 1.0/(base**(torch.arange(0, head_dim, 2).float()/head_dim))
  seq = torch.arange(max_seq_len)
  freq = torch.outer(seq, theta)
  return torch.cos(freq), torch.sin(freq)

def apply_rope(x, cos, sin):
  dim = x.size(-1)//2
  x1, x2 = x[..., :dim], x[..., dim:]
  y1 = x1*cos - x2*sin
  y2 = x1*sin + x2*cos
  return torch.cat([y1, y2], dim=-1)

def gather_seq(x, group):
  b, s, d = x.size()
  world_size = dist.get_world_size(group)
  x_gathered = torch.empty((s*world_size, b, d), dtype=x.dtype, device=x.device)
  dist.all_gather_into_tensor(x_gathered, x.transpose(0,1).contiguous(), group=group)
  return x_gathered.transpose(0,1)

def reduce_scatter_seq(x, group):
  b, s, d = x.size()
  world_size = dist.get_world_size(group)
  x_scattered = torch.empty((s//world_size, b, d), dtype=x.dtype, device=x.device)
  dist.reduce_scatter_into_tensor(x_scattered, x.transpose(0,1).contiguous(), op=dist.ReduceOp.SUM, group=group)
  return x_scattered.transpose(0,1)

def flash_attn(q, k, v, causal, block_size=128):
  b, n_heads, s, head_dim = q.size()
  scale = 1./(head_dim ** 0.5)
  if k.size(1) != n_heads:
    k = k.repeat_interleave(n_heads//k.size(1), dim=1)
    v = v.repeat_interleave(n_heads//k.size(1), dim=1)
  if s % block_size:
    to_pad = block_size - s % block_size
    q = F.pad(q, (0, 0, 0, 0, 0, to_pad))
    k = F.pad(k, (0, 0, 0, 0, 0, to_pad))
    v = F.pad(v, (0, 0, 0, 0, 0, to_pad))
  vo = torch.empty((b, n_heads, s, head_dim), dtype=q.dtype, device=q.device)
  q_end = (s+block_size-1)//block_size
  for i in range(q_end):
    qi = q[:, :, block_size*i : block_size*(i+1)]
    m = torch.full((b, n_heads, block_size, 1), torch.finfo(q.dtype).min, dtype=q.dtype, device=q.device)
    d = torch.zeros_like(m)
    o = torch.zeros((b, n_heads, block_size, head_dim), dtype=q.dtype, device=q.device)
    k_end = i+1 if causal else q_end
    for j in range(k_end):
      kj = k[:, :, block_size*j : block_size*(j+1)]
      vj = v[:, :, block_size*j : block_size*(j+1)]
      logits = (qi @ kj.transpose(-1,-2)) * scale
      if block_size*(j+1) > s:
        logits[:, :, :, s%block_size:] = torch.finfo(logits.dtype).min
      if i==j and causal:
        mask = torch.ones((block_size, block_size), device=logits.device).triu(diagonal=1).bool()
        logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0), torch.finfo(logits.dtype).min)
      m_new = torch.maximum(m, logits.amax(dim=-1, keepdim=True))
      alpha = torch.exp(m-m_new)
      beta = torch.exp(logits-m_new)
      d = alpha*d + beta.sum(dim=-1, keepdim=True)
      o = alpha*o + beta @ vj
      m = m_new
    vo_end = min(s, block_size*(i+1))
    vo[:, :, block_size*i:vo_end].copy_((o/d)[:, :, :vo_end-block_size*i])
  return vo.contiguous().view(b, s, -1)

def send_recv(xs, group):
  world_size = dist.get_world_size(group)
  rank = dist.get_rank(group)
  bufs = [x.clone() for x in xs]
  ops = []
  for x, buf in zip(xs, bufs):
    if rank>0:
      ops.append(dist.P2POp(dist.irecv, buf, rank-1, group))
    if rank+1<world_size:
      ops.append(dist.P2POp(dist.isend, x, rank+1, group))
  ops = dist.batch_isend_irecv(ops) if ops else []
  return ops, bufs

def ring_attn(q, k, v, group):
  b, n_heads, s, head_dim = q.size()
  scale = 1./(head_dim**0.5)
  n_kv_heads = k.size(1)
  rank = dist.get_rank(group)
  m = torch.full((b, n_heads, s, 1), torch.finfo(q.dtype).min, dtype=q.dtype, device=q.device)
  d = torch.zeros_like(m)
  o = torch.zeros((b, n_heads, s, head_dim), dtype=q.dtype, device=q.device)
  for i in range(rank+1):
    ring_k, ring_v = k.clone(), v.clone()
    ops, bufs = send_recv([k, v], group)
    if n_heads != n_kv_heads:
      ring_k = ring_k.repeat_interleave(n_heads//n_kv_heads, dim=1)
      ring_v = ring_v.repeat_interleave(n_heads//n_kv_heads, dim=1)
    logits = (q @ ring_k.transpose(-1,-2)) * scale
    if i==0:
      mask = torch.ones((s, s), device=logits.device).triu(diagonal=1).bool()
      logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0), torch.finfo(logits.dtype).min)
    m_new = torch.maximum(m, logits.amax(dim=-1, keepdim=True))
    alpha = torch.exp(m-m_new)
    beta = torch.exp(logits-m_new)
    d = alpha*d + beta.sum(dim=-1, keepdim=True)
    o = alpha*o + beta @ ring_v
    for op in ops:
      op.wait()
    k, v = bufs
  return (o/d).contiguous().view(b, s, -1)

class MultiHeadSelfAttention(nn.Module):
  def __init__(self, dim, n_heads, n_kv_heads=None, dropout=0.0, use_bias=True, tp_group=None, use_sp=True, cp_group=None, use_flash=False):
    super().__init__()
    self.n_heads = n_heads
    self.n_kv_heads = n_kv_heads or n_heads
    self.head_dim = dim // n_heads
    self.tp_group = tp_group
    self.use_sp = use_sp
    self.cp_group = cp_group
    self.use_flash = use_flash
    if self.tp_group:
      world_size = dist.get_world_size(self.tp_group)
      self.n_heads //= world_size
      self.n_kv_heads //= world_size
    self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=use_bias)
    self.kv_proj = nn.Linear(dim, 2*self.n_kv_heads * self.head_dim, bias=use_bias)
    self.out_proj = nn.Linear(self.n_heads*self.head_dim, dim, bias=use_bias)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x, cos, sin, causal, kv_cache):
    assert not (kv_cache is not None and self.cp_group is not None), "context parallelism doesn't support kv_cache."
    assert not (not causal and self.cp_group is not None), "context parallelism doesn't support non-causal."
    assert not (kv_cache is not None and self.use_flash), "flash attn doesn't support kv_cache."
    if self.tp_group and self.use_sp:
      x = gather_seq(x, self.tp_group)
    b, s, _ = x.size()
    q = self.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1,2)
    curr_k, curr_v = self.kv_proj(x).view(b, s, -1).chunk(2, dim=-1)
    curr_k = curr_k.view(b, s, self.n_kv_heads, self.head_dim).transpose(1,2)
    curr_v = curr_v.view(b, s, self.n_kv_heads, self.head_dim).transpose(1,2)
    q, curr_k = apply_rope(q, cos, sin), apply_rope(curr_k, cos, sin)
    if kv_cache is not None:
      k_cache, v_cache = kv_cache
      k = torch.cat([k_cache, curr_k], dim=2)
      v = torch.cat([v_cache, curr_v], dim=2)
    else:
      k, v = curr_k, curr_v
    kv_cache = k, v
    if self.cp_group:
      vo = ring_attn(q, k, v, self.cp_group)
    elif self.use_flash:
      vo = flash_attn(q, k, v, causal)
    else:
      if self.n_heads != self.n_kv_heads:
        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
      logits = (q @ k.transpose(-1,-2)) / (self.head_dim ** 0.5)
      if causal:
        ks = k.size(2)
        mask = torch.ones((s, ks), device=logits.device).triu(diagonal=ks-s+1).bool()
        logits = logits.masked_fill(mask.unsqueeze(0).unsqueeze(0), torch.finfo(logits.dtype).min)
      scores = self.dropout(F.softmax(logits, dim=-1))
      vo = (scores @ v).contiguous().view(b, s, -1)
    out = self.out_proj(vo)
    if self.tp_group:
      if self.use_sp:
        out = reduce_scatter_seq(out, self.tp_group)
      else:
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.tp_group)
    return out, kv_cache

class MLP(nn.Module):
  def __init__(self, dim, multiplier, dropout=0.0, use_bias=True, tp_group=None, use_sp=False):
    super().__init__()
    up_dim = dim * multiplier
    self.tp_group = tp_group
    self.use_sp = use_sp
    if self.tp_group:
      world_size = dist.get_world_size(self.tp_group)
      up_dim //= world_size
    self.up = nn.Linear(dim, up_dim, bias=use_bias)
    self.gate = nn.Linear(dim, up_dim, bias=use_bias)
    self.down = nn.Linear(up_dim, dim, bias=use_bias)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    if self.tp_group and self.use_sp:
      x = gather_seq(x, self.tp_group)
    x = F.silu(self.gate(x)) * self.up(x)
    x = self.down(x)
    if self.tp_group:
      if self.use_sp:
        x = reduce_scatter_seq(x, self.tp_group)
      else:
        dist.all_reduce(x, op=dist.ReduceOp.SUM, group=self.tp_group)
    x = self.dropout(x)
    return x

class LayerNorm(nn.Module):
  def __init__(self, dim, eps):
    super().__init__()
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(dim))
    self.beta = nn.Parameter(torch.zeros(dim))

  def forward(self, x):
    return self.gamma * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True, unbiased=False) + self.eps) + self.beta

class RmsNorm(nn.Module):
  def __init__(self, dim, eps):
    super().__init__()
    self.eps = eps
    self.gamma = nn.Parameter(torch.ones(dim))

  def forward(self, x):
    return self.gamma * x * torch.rsqrt(torch.mean(x**2, dim=-1, keepdim=True) + self.eps)


@dataclass
class ModelConfig:
  num_layers: int
  vocab_size: int
  max_seq_len: int
  model_dim: int
  n_heads: int
  n_kv_heads: int | None
  attn_dropout: float
  ffn_multiplier: int
  ffn_dropout: float
  norm_type: str
  norm_eps: float
  use_bias: bool
  use_flash: bool

@dataclass
class ModelParal:
  tp_group: dist.ProcessGroup | None
  use_sp: bool
  cp_group: dist.ProcessGroup | None

@dataclass
class SamplingConfig:
  max_gen_len: int
  temperature: float
  topk: int | None
  topp: float | None
  sos_id: int | None
  eos_id: int | None


class TransformerBlock(nn.Module):
  def __init__(self, config: ModelConfig, mp: ModelParal):
    super().__init__()
    self.mhsa = MultiHeadSelfAttention(
        config.model_dim,
        config.n_heads,
        config.n_kv_heads,
        config.attn_dropout,
        config.use_bias,
        mp.tp_group,
        mp.use_sp,
        mp.cp_group,
        config.use_flash,
    )
    Norm = RmsNorm if config.norm_type=="rms" else LayerNorm
    self.norm1 = Norm(config.model_dim, config.norm_eps)
    self.ffn = MLP(
        config.model_dim,
        config.ffn_multiplier,
        config.ffn_dropout,
        config.use_bias,
        mp.tp_group,
        mp.use_sp,
    )
    self.norm2 = Norm(config.model_dim, config.norm_eps)

  def forward(self, x, cos, sin, causal, kv_cache):
    attn_x, kv_cache = self.mhsa(self.norm1(x), cos, sin, causal, kv_cache)
    x = x + attn_x
    x = x + self.ffn(self.norm2(x))
    return x, kv_cache


class Transformer(nn.Module):
  def __init__(self, config: ModelConfig, mp: ModelParal):
    super().__init__()
    self.config = config
    self.mp = mp
    self.embed = nn.Embedding(config.vocab_size, config.model_dim)
    head_dim = config.model_dim // config.n_heads
    cos, sin = generate_rope(config.max_seq_len, head_dim)
    self.register_buffer("cos", cos)
    self.register_buffer("sin", sin)
    self.layers = nn.ModuleList([
        TransformerBlock(config, mp) for _ in range(config.num_layers)
    ])
    Norm = RmsNorm if config.norm_type=="rms" else LayerNorm
    self.norm = Norm(config.model_dim, config.norm_eps)
    self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)

  def forward(self, x, causal, kv_cache):
    x = self.embed(x)
    start_pos = 0
    if self.mp.tp_group and self.mp.use_sp:
      world_size = dist.get_world_size(self.mp.tp_group)
      rank = dist.get_rank(self.mp.tp_group)
      per_rank_seq = x.size(1)//world_size
      x = x[:, rank*per_rank_seq:(rank+1)*per_rank_seq]
      start_pos = rank*per_rank_seq
    if kv_cache is not None:
      start_pos += kv_cache[0][0].size(2)
    else:
      kv_cache = [None] * len(self.layers)
    cos = self.cos[start_pos:start_pos+x.size(1)].unsqueeze(0).unsqueeze(0)
    sin = self.sin[start_pos:start_pos+x.size(1)].unsqueeze(0).unsqueeze(0)
    new_kv_cache = []
    for i, layer in enumerate(self.layers):
      x, cache = layer(x, cos, sin, causal, kv_cache[i])
      new_kv_cache.append(cache)
    x = self.norm(x)
    if self.mp.tp_group and self.mp.use_sp:
      x = gather_seq(x, self.mp.tp_group)
    x = self.lm_head(x)
    return x, new_kv_cache

  def _sample_next_token(self, logits, sampling_config):
    if logits.ndim==3:
      logits = logits[:, -1, :]
    assert sampling_config.temperature > 0
    logits = logits / sampling_config.temperature
    if sampling_config.topk is not None:
      topk_logits, _ = torch.topk(logits, sampling_config.topk)
      logits[logits<topk_logits[:, [-1]]] = torch.finfo(logits.dtype).min
    probs = F.softmax(logits, dim=-1)
    if sampling_config.topp is not None:
      sorted_probs, sorted_indices = torch.sort(probs, descending=True)
      cumsum_probs = torch.cumsum(sorted_probs, dim=-1)
      to_mask = cumsum_probs - sorted_probs > sampling_config.topp
      sorted_probs[to_mask] = 0
      probs = torch.zeros_like(probs).scatter_(1, sorted_indices, sorted_probs)
      probs = probs / probs.sum(dim=-1, keepdim=True).clamp(1e-6)
    return torch.multinomial(probs, num_samples=1)

  def generate(self, tokens, sampling_config: SamplingConfig):
    if tokens.ndim == 1:
      tokens = tokens.unsqueeze(0)
    if tokens.numel() == 0:
      assert sampling_config.sos_id is not None
      tokens = torch.tensor([[sampling_config.sos_id]], dtype=tokens.dtype, device=tokens.device)
    logits, kv_cache = self(tokens, causal=True, kv_cache=None)
    next_token = self._sample_next_token(logits, sampling_config)
    generated_tokens = [next_token]
    for _ in range(sampling_config.max_gen_len):
      if sampling_config.eos_id is not None and (next_token==sampling_config.eos_id).all():
        break
      logits, kv_cache = self(next_token, causal=True, kv_cache=kv_cache)
      next_token = self._sample_next_token(logits, sampling_config)
      generated_tokens.append(next_token)
    return torch.cat(generated_tokens, dim=-1)
