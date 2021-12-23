
class CustomOptim():
    "Optim wrapper that implement cycling learning rates"
    def __init__(self, factor, warmup, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor

    def step(self, epoch):
        rate = self.rate(epoch)
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self.optimizer.step()

    def rate(self, epoch):
        self._step += 1
        if (epoch) % 50 == 0 and epoch > 0:
            self._step = self.warmup -1
        return self.factor * min(1.0, self._step / self.warmup) / max(self._step, self.warmup)

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def state_dict(self):
        return self.optimizer.state_dict()

    def _get_lr(self):
        return self.optimizer.param_groups[0]['lr']

