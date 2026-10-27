import torch
import gpytorch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from linear_operator.operators import DiagLinearOperator as DiagMatrix
from olgp.gpp.kernels import ScaleKernel, GaussianKernel
from gpytorch.priors import NormalPrior

class NeuralNoiseCovar(gpytorch.likelihoods.noise_models.Noise):
    def __init__(self, cfg, embedding_dim, d, fun_dim, min_noise=1e-3):
        super().__init__()
        self.cfg = cfg
        self.d = d
        self.fun_dim = fun_dim
        self.min_noise = min_noise

        self.noise_net = nn.Sequential(
            nn.LayerNorm(self.d + fun_dim + embedding_dim),
            spectral_norm(nn.Linear(self.d + fun_dim + embedding_dim, 64)),
            nn.GELU(),
            spectral_norm(nn.Linear(64, 32)),
            nn.GELU(),
            spectral_norm(nn.Linear(32, 1)),
            nn.Softplus()
        )
        # Initialize last linear bias to give small initial noise
        with torch.no_grad():
            for layer in reversed(self.noise_net):
                if isinstance(layer, nn.Linear):
                    layer.bias.fill_(-3.0)
                    break

        self.last_x = None

    def forward(self, *params, shape=None, **kwargs):
        # GPyTorch passes the inputs when building the prediction cache; otherwise
        # use the inputs stored by REEFGP.__call__
        if len(params) > 0 and params[0] is not None:
            x = params[0]
        elif self.last_x is not None:
            x = self.last_x
        else:
            raise RuntimeError("NeuralNoiseCovar received no inputs; call it through REEFGP.")

        # Unwrap nested lists/tuples from GPyTorch
        while isinstance(x, (tuple, list)):
            x = x[0]

        # All input columns except the backbone prediction
        embeddings = x[..., : -1]
        noise = self.noise_net(embeddings).squeeze(-1) + self.min_noise
        return DiagMatrix(noise)

class NNHeteroscedasticLikelihood(gpytorch.likelihoods._GaussianLikelihoodBase):
    def __init__(self, cfg, embedding_dim, d, fun_dim):
        super().__init__(noise_covar=NeuralNoiseCovar(cfg, embedding_dim, d, fun_dim))

class FlowNetwork(nn.Module):
    def __init__(self, cfg, input_dim=384, hidden_dim=128, latent_dim=64):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            spectral_norm(nn.Linear(input_dim, hidden_dim)),
            nn.ReLU(),
            spectral_norm(nn.Linear(hidden_dim, latent_dim))
        )
    def forward(self, x):
        return self.net(x)

class NeuralMean(gpytorch.means.Mean):
    def __init__(self, input_idx):
        super().__init__()
        self.input_idx = input_idx # Index of the NN prediction in x

    def forward(self, x):
        return x[..., self.input_idx]

class REEFGP(gpytorch.models.ExactGP):
    def __init__(self, cfg, problem_name, train_x, train_y, likelihood, embedding_dim=128, d=2, fun_dim=0):
        super().__init__(train_x, train_y, likelihood)
        self.cfg = cfg
        self.problem_name = problem_name
        self.d = d
        self.fun_dim = fun_dim

        # Input layout: [coords (d), input functions (fun_dim), embeddings, nn_pred (1)]
        self.mean_module = NeuralMean(input_idx=-1)

        # Spectral-normalized warp of the embeddings
        self.latent_dim = 64
        self.flow = FlowNetwork(cfg, input_dim=embedding_dim, hidden_dim=128, latent_dim=self.latent_dim)

        self.kernel_class = GaussianKernel

        self.k_spatial = self.kernel_class(
                ard_num_dims=d,
                active_dims=list(range(d)),
                lengthscale_prior=NormalPrior(loc=-1.5, scale=0.5),
            )
        self.k_fun = self.kernel_class(
                ard_num_dims=fun_dim,
                active_dims=list(range(d, d + fun_dim)) if fun_dim > 0 else [],
                lengthscale_prior=NormalPrior(loc=0.0, scale=0.5)
            ) if fun_dim > 0 else None

        self.k_embedding = self.kernel_class(
                ard_num_dims=self.latent_dim,
                active_dims=list(range(d + fun_dim, d + fun_dim + self.latent_dim)),
                lengthscale_prior=NormalPrior(loc=-1.0, scale=0.5),
            )

        # Product kernel
        if fun_dim > 0:
            self.covar_module = ScaleKernel(self.k_spatial * self.k_fun * self.k_embedding)
        else:
            self.covar_module = ScaleKernel(self.k_spatial * self.k_embedding)

        self.covar_module.register_prior(
            "outputscale_prior",
            NormalPrior(loc=0.0, scale=0.5),  # in log10 space; mode at scale=1
            "outputscale"
        )

    def __call__(self, *args, **kwargs):  # pass the inputs to the heteroscedastic noise network
        if len(args) > 0 and hasattr(self.likelihood, "noise_covar"):
            self.likelihood.noise_covar.last_x = args[0]
        return super().__call__(*args, **kwargs)

    def forward(self, x):
        coords = x[..., :self.d]
        if self.fun_dim > 0:
            fun = x[..., self.d : self.d + self.fun_dim]
        embeddings = x[..., self.fun_dim + self.d : -1]
        projected_embeddings = self.flow(embeddings)

        if self.fun_dim > 0:
            flow_x = torch.cat([coords, fun, projected_embeddings], dim=-1)
        else:
            flow_x = torch.cat([coords, projected_embeddings], dim=-1)

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(flow_x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
