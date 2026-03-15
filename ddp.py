import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from dataclasses import dataclass, field

### example usage
# model = Transformer(config, mp).to(device)
# param_buckets = ParamBuckets(model.parameters(), bucket_mb_size, dp_group)
# criteria = nn.CrossEntropyLoss()
# optimizer = optim.AdamW(stage.parameters(), lr=1e-4)

# model.train()
# for x, y in dataloader:
#   x, y = x.to(device), y.to(device)              
#   yhat, _ = model(x)
#   x = x.view(-1, config.vocab_size)
#   y = y.view(-1)
#   loss = criteria(x, y)
#   loss.backward()  # bucket hook happens
#   param_buckets.update()  # call bucket to udpate grads  
#   optimizer.step()
#   optimizer.zero_grad()

@dataclass
class Bucket:
  params: list[nn.Parameter] = field(default_factory=list)
  total_bytes: int = 0
  pending: int = 0
  finalized: bool = False
  buffer: torch.Tensor | None = None
  async_op: dist.Work | None = None

  def finalize(self):
    self.finalized = True
    self.pending = len(self.params)
    self.buffer = torch.zeros(sum(p.numel() for p in self.params), dtype=self.params[0].dtype, device=self.params[0].device)
    self.async_op = None

  def add_param(self, p):
    assert not self.finalized, "can't add param after finalized"
    self.params.append(p)
    self.total_bytes += p.numel() * p.element_size()

  def size_mb(self):
    return float(self.total_bytes) / (1024**2)

  def ready(self):
    return self.pending == 0

  def flush(self, group):
    offset = 0
    for p in self.params:
      self.buffer[offset:offset+p.numel()].copy_(p.grad.view(-1))
      offset += p.numel()
    self.async_op = dist.all_reduce(self.buffer, op=dist.ReduceOp.AVG, group=group, async_op=True)

  def _reset(self):
    self.pending = len(self.params)
    self.async_op = None

  def update(self):
    assert self.async_op is not None, "bucket not flushed when backward!"
    self.async_op.wait()
    offset = 0
    for p in self.params:
      p.grad.copy_(self.buffer[offset:offset+p.numel()].view_as(p))
      offset += p.numel()
    self._reset()


class ParamBuckets:
  def __init__(self, params, bucket_mb_size, dp_group):
    self.params = [p for p in params if p.requires_grad]
    self.bucket_mb_size = bucket_mb_size
    self.dp_group = dp_group
    self.buckets = []
    self._build_buckets()
    self._add_hook()

  def _build_buckets(self):
    bkt = Bucket()
    for p in reversed(self.params):
      bkt.add_param(p)
      if bkt.size_mb() > self.bucket_mb_size:
        bkt.finalize()
        self.buckets.append(bkt)
        bkt = Bucket()
    if bkt.total_bytes > 0:
      bkt.finalize()
      self.buckets.append(bkt)

  def _add_hook(self):

    def make_hook(bkt):
      def hook(p):
        bkt.pending -= 1
        if bkt.ready():
          bkt.flush(self.dp_group)
      return hook

    for bkt in self.buckets:
      for p in bkt.params:
        p.register_post_accumulate_grad_hook(make_hook(bkt))

  def update(self):
    for bkt in self.buckets:
      bkt.update()
