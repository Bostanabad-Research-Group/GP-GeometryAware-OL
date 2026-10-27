import torch
from gpytorch.kernels import Kernel

class ScaleKernel(Kernel):
    def __init__(self, base_kernel, batch_shape=None, **kwargs):
        super().__init__(**kwargs) # super(Kernel, self).__init__() # bypass Kernel.__init__ and go to Module
        self.base_kernel = base_kernel
        self._batch_shape = torch.Size([]) if batch_shape is None else batch_shape

        # Register raw log_scale (log10 space)
        self.register_parameter(
            name="outputscale",
            parameter=torch.nn.Parameter(torch.zeros(*self._batch_shape, 1, 1))
        )

    def forward(self, x1, x2, diag=False, **params):
        # Calculate the multiplier: 10^log_scale
        scale = torch.pow(10, self.outputscale)

        # Get the underlying covariance
        res = self.base_kernel(x1, x2, diag=diag, **params)

        # Apply scaling
        if diag:
            return res * scale.view(*self._batch_shape, -1)
        return res * scale
