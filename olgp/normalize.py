
class StandardScaler:
    def __init__(self, mean=None, std=None, device=None):
        self.mean = mean
        self.std = std
        if device is not None:
            if self.mean is not None:
                self.mean = self.mean.to(device)
            if self.std is not None:
                self.std = self.std.to(device)

    def fit_transform(self, x):
        # Per-feature statistics over all samples and points: [B, N, F] -> [1, 1, F]
        if x.dim() > 1:
            reduce_dims = list(range(x.dim() - 1))
        else:
            reduce_dims = [0]

        self.mean = x.mean(dim=reduce_dims, keepdim=True)
        self.std = x.std(dim=reduce_dims, unbiased=False, keepdim=True)

        # Constant features keep a unit scale
        self.std[self.std < 1e-8] = 1.0

        x_norm = (x - self.mean) / self.std

        return x_norm, self.mean, self.std

    def transform(self, x):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted.")

        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)

        return (x - self.mean) / self.std

    def inverse(self, x_norm):
        if self.mean is None or self.std is None:
            raise ValueError("Scaler has not been fitted.")

        if self.mean.device != x_norm.device:
            self.mean = self.mean.to(x_norm.device)
            self.std = self.std.to(x_norm.device)

        return (x_norm * self.std) + self.mean
