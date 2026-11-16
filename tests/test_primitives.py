import pytest
import torch
import torch.nn as nn

from config import TrainConfig
from primitives import (VARIANTS, EMLLayer, KANLayer, ReLULayer, build_network,
                        count_params, iso_hidden)

CFG = TrainConfig()


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("in_dim", [2, 3])
def test_forward_shape_and_finite(variant, in_dim):
    torch.manual_seed(0)
    net = build_network(variant, in_dim, CFG)
    y = net(torch.randn(64, in_dim))
    assert y.shape == (64, 1)
    assert torch.isfinite(y).all()


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_output_layer_is_linear(variant):
    # Régression : l'ancienne version mettait un ReLU en sortie de ReLU-MLP,
    # incapable de prédire une valeur négative.
    net = build_network(variant, 2, CFG)
    assert isinstance(net[-1], nn.Linear)


def test_relu_mlp_can_output_negative_values():
    torch.manual_seed(0)
    net = build_network("ReLU-MLP", 2, CFG)
    with torch.no_grad():
        net[-1].bias.fill_(-10.0)
    assert (net(torch.randn(32, 2)) < 0).any()


@pytest.mark.parametrize("variant", list(VARIANTS))
@pytest.mark.parametrize("in_dim", [2, 3])
def test_iso_parametric_within_10_percent(variant, in_dim):
    ref = count_params(build_network("ReLU-MLP", in_dim, CFG))
    n = count_params(build_network(variant, in_dim, CFG))
    assert abs(n - ref) / ref < 0.10, f"{variant}: {n} vs {ref}"


def test_iso_hidden_reference_is_unchanged():
    assert iso_hidden("ReLU-MLP", 2, 24, 3) == 24


def test_eml_ablations_have_no_dead_parameters():
    full = EMLLayer(4, 5)
    noexp = EMLLayer(4, 5, use_exp=False)
    nolog = EMLLayer(4, 5, use_log=False)
    assert count_params(noexp) == count_params(nolog) < count_params(full)
    with pytest.raises(ValueError):
        EMLLayer(4, 5, use_exp=False, use_log=False)


def test_eml_formula():
    torch.manual_seed(0)
    layer = EMLLayer(3, 4, delta=1e-3)
    x = torch.randn(8, 3)
    expected = (layer.alpha * torch.exp(layer.W_exp(x).clamp(-6, 6))
                - layer.beta * torch.log(layer.W_log(x).abs() + 1e-3) + layer.bias)
    assert torch.allclose(layer(x), expected)


def test_smooth_log_has_bounded_gradient():
    z = torch.linspace(-1e-3, 1e-3, 101, requires_grad=True)
    abs_log = EMLLayer(1, 1, log_mode="abs").log_branch(z).sum()
    (g_abs,) = torch.autograd.grad(abs_log, z)
    smooth = EMLLayer(1, 1, log_mode="smooth").log_branch(z).sum()
    (g_smooth,) = torch.autograd.grad(smooth, z)
    assert g_smooth.abs().max() <= 0.5 + 1e-6
    assert g_abs.abs().max() > 100      # la version "abs" explose près de 0


class _KANLoop(nn.Module):
    """Implémentation de référence par double boucle."""

    def __init__(self, kan: KANLayer):
        super().__init__()
        self.kan = kan

    def forward(self, x):
        k, g = self.kan, self.kan.grid
        out = torch.zeros(x.shape[0], k.d_out)
        for j in range(k.d_out):
            for i in range(k.d_in):
                a = j * k.d_in + i
                xn = (k.scale[a] * (x[:, i] - k.shift[a])).clamp(g[0], g[-1] - 1e-6)
                idx = torch.searchsorted(g, xn.contiguous()).clamp(1, k.k - 1)
                t = (xn - g[idx - 1]) / (g[idx] - g[idx - 1])
                out[:, j] += k.ctrl[a, idx - 1] * (1 - t) + k.ctrl[a, idx] * t
        return out


@pytest.mark.parametrize("d_in,d_out", [(1, 1), (2, 3), (3, 4), (5, 2)])
@pytest.mark.parametrize("batch", [1, 8, 256])
def test_kan_vectorized_matches_loop(d_in, d_out, batch):
    torch.manual_seed(42)
    kan = KANLayer(d_in, d_out)
    with torch.no_grad():
        kan.scale.uniform_(0.5, 2.0)
        kan.shift.uniform_(-1, 1)
    x = 3 * torch.randn(batch, d_in)
    assert torch.allclose(kan(x), _KANLoop(kan)(x), atol=1e-6)
