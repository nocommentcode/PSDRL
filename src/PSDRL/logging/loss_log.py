class LossLog:
    def __init__(self, name):
        self.n = 0
        self.total = 0
        self.name = name

    def __add__(self, other):
        self.n += 1
        self.total += other if type(other) == int else other.item()
        return self

    def __iadd__(self, other):
        self.n += 1
        self.total += (
            other if type(other) == int or type(other) == float else other.item()
        )
        return self

    def get_scalar(self):
        return (
            [f"Loss/{self.name}-Total", f"Loss/{self.name}-Average"],
            [self.total, self.total / self.n],
        )
