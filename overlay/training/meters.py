from __future__ import annotations


class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.value = 0.0
        self.average = 0.0
        self.total = 0.0
        self.count = 0

    def update(self, value: float, count: int = 1) -> None:
        self.value = float(value)
        self.total += float(value) * count
        self.count += int(count)
        self.average = self.total / max(self.count, 1)
