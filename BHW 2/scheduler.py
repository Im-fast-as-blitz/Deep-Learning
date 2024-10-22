import torch


class MyScheduler:
    def __init__(self, optimizer, warm, init_lr, end_lr, last_epoches):
        self.optimizer = optimizer
        self.warm = warm
        self.init_lr = init_lr
        self.end_lr = end_lr
        self.last_epoches = last_epoches
        self._rate = init_lr
        self._q = (end_lr / init_lr) ** (1 / warm)

        self._step = 0
        self._new_sched = None

    def step(self):
        self._step += 1
        if self._step <= self.warm:
            if self._step == self.warm:
                self._rate = self.end_lr
            else:
                self._rate *= self._q
            for group in self.optimizer.param_groups:
                group['lr'] = self._rate
        else:
            if self._new_sched is None:
                self._new_sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.last_epoches, eta_min=self.init_lr)
            self._new_sched.step()
