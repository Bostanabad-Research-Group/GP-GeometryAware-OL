import torch
from gpytorch.constraints import Interval


class SoftClamp(Interval):
    """
    Identity core + exponential tails.
    lower_bound=a, upper_bound=b, margin=ε with 0 < ε < (b-a)/2
    """

    def __init__(self, lower_bound, upper_bound, margin: float = 1e-2, initial_value=None):
        # Let Interval validate/store bounds & initial_value
        super().__init__(lower_bound, upper_bound, transform=None, inv_transform=None, initial_value=None)

        # Bounds and margin must be scalars
        for name, val in [("lower_bound", lower_bound), ("upper_bound", upper_bound), ("margin", margin)]:
            t = torch.as_tensor(val)
            if t.ndim != 0:
                raise ValueError(f"{name} must be a scalar (got shape {tuple(t.shape)}).")

        # Prepare margin as a buffer on the right dtype/device
        eps = torch.as_tensor(margin, dtype=self.lower_bound.dtype, device=self.lower_bound.device)

        # Require 0 < ε < (b - a)/2
        width = self.upper_bound - self.lower_bound
        tiny_eps = torch.finfo(self.lower_bound.dtype).eps
        if not torch.all(eps > 0):
            raise ValueError(f"margin must be > 0 (got {margin}).")
        if not torch.all(eps < (width / 2) - tiny_eps):
            raise ValueError(
                f"margin must be < (upper_bound - lower_bound)/2. "
                f"Got margin={margin}, width={width.item() if width.numel() == 1 else 'tensor'}."
            )

        self.register_buffer("margin", eps)

        if initial_value is not None:
            self._initial_value = self.inverse_transform(
                torch.as_tensor(initial_value, dtype=self.lower_bound.dtype, device=self.lower_bound.device)
            )

    def transform(self, raw_tensor: torch.Tensor) -> torch.Tensor:
        """
        Map unconstrained raw values into the interval (a, b) with
        exponential tails and identity in the middle.
        """
        # a = lower bound, b = upper bound, ε = margin
        # x_raw is the unconstrained value; transform maps x_raw -> x_act in (a, b)
        a, b, eps = self.lower_bound, self.upper_bound, self.margin
        xL, xR = a + eps, b - eps
        out = raw_tensor.clone()  # keeps autograd link

        # Piecewise map (x_raw -> x_act):
        #   if x_raw < xL:     x_act = a + ε * exp((x_raw - xL)/ε)
        #   if xL <= x_raw <= xR: x_act = x_raw               (identity in the core)
        #   if x_raw > xR:     x_act = b - ε * exp(-(x_raw - xR)/ε)

        mask_left = raw_tensor < xL
        mask_right = raw_tensor > xR

        out[mask_left] = a + eps * torch.exp((raw_tensor[mask_left] - xL) / eps)
        out[mask_right] = b - eps * torch.exp(-(raw_tensor[mask_right] - xR) / eps)

        return out

    def inverse_transform(self, transformed_tensor: torch.Tensor) -> torch.Tensor:
        """
        Inverse of the transform: map (a, b) values back to unconstrained raw space.
        """
        # Piecewise inverse (x_act -> x_raw), with xL = a + ε and xR = b - ε:
        #   if x_act < xL:          x_raw = xL + ε * log((x_act - a)/ε)
        #   if xL <= x_act <= xR:   x_raw = x_act
        #   if x_act > xR:          x_raw = xR - ε * log((b - x_act)/ε)
        a, b, eps = self.lower_bound, self.upper_bound, self.margin
        xL, xR = a + eps, b - eps
        out = transformed_tensor.clone()

        # Clamp the log arguments to machine epsilon to avoid log(0)
        tiny = torch.finfo(transformed_tensor.dtype).eps

        mask_left = transformed_tensor < xL
        mask_right = transformed_tensor > xR

        out[mask_left] = xL + eps * torch.log(((transformed_tensor[mask_left] - a) / eps).clamp_min(tiny))
        out[mask_right] = xR - eps * torch.log(((b - transformed_tensor[mask_right]) / eps).clamp_min(tiny))

        return out

    def __repr__(self):
        def fmt(t: torch.Tensor) -> str:
            if t.numel() == 1:
                return f"{t.item():.6g}"
            # short 1-D vectors inline, otherwise only shape and dtype
            if t.ndim == 1 and t.numel() <= 8:
                return "[" + ", ".join(f"{v:.6g}" for v in t.tolist()) + "]"
            return f"tensor(shape={tuple(t.shape)}, dtype={t.dtype})"

        return f"SoftClamp({fmt(self.lower_bound)}, {fmt(self.upper_bound)}, margin={fmt(self.margin)})"
