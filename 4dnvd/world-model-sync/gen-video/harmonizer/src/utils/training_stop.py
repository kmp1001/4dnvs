"""Coordinate graceful stops at completed optimizer steps across DDP ranks."""

import signal

import torch
import torch.distributed as dist


class TrainingStop:
    def __init__(self, device):
        self.device = device
        self.signum = 0
        self.previous = {
            number: signal.signal(number, self._request)
            for number in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1)
        }

    def _request(self, signum, frame):
        self.signum = signum

    def requested(self):
        number = torch.tensor(self.signum, dtype=torch.int32, device=self.device)
        if dist.is_initialized():
            dist.all_reduce(number, op=dist.ReduceOp.MAX)
        return number.item()

    def close(self):
        for number, handler in self.previous.items():
            signal.signal(number, handler)
