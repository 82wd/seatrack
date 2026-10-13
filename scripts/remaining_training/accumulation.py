"""Accumulate equally sized microbatches before clipping and stepping."""
import torch

class Accumulation:
    def __init__(self, optimizer, scaler, parameters, steps, clip):
        assert steps >= 1
        self.optimizer, self.scaler = optimizer, scaler
        self.parameters = list(parameters)
        self.steps, self.clip, self.pending, self.updates = steps, clip, 0, 0
        optimizer.zero_grad(set_to_none=True)

    def backward(self, loss):
        self.scaler.scale(loss / self.steps).backward()
        self.pending += 1
        if self.pending != self.steps:
            return False
        if self.clip > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.parameters, self.clip)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.pending = 0
        self.updates += 1
        return True

    def finish(self):
        assert self.pending == 0, 'Incomplete accumulation group at epoch boundary'
