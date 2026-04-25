"""
test_kan_vectorization.py — verifie que KANLayer vectorise donne
les memes resultats que l'ancienne implementation par boucle.

Usage : python test_kan_vectorization.py
"""

import torch
import torch.nn as nn
from primitives import KANLayer, BSplineActivation


class KANLayer_loop(nn.Module):
    """Ancienne implementation avec double boucle Python (reference)."""

    def __init__(self, in_features, out_features,
                 k=8, x_min=-6.0, x_max=6.0, **_):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activations = nn.ModuleList([
            BSplineActivation(k, x_min, x_max)
            for _ in range(in_features * out_features)
        ])

    def forward(self, x):
        B = x.shape[0]
        out = torch.zeros(B, self.out_features, device=x.device)
        for j in range(self.out_features):
            for i in range(self.in_features):
                act = self.activations[j * self.in_features + i]
                out[:, j] = out[:, j] + act(x[:, i])
        return out


def copy_weights(loop_layer: KANLayer_loop, vec_layer: KANLayer):
    """Copie les poids de la version boucle vers la version vectorisee."""
    with torch.no_grad():
        for a_idx, act in enumerate(loop_layer.activations):
            vec_layer.ctrl[a_idx]  = act.ctrl.data
            vec_layer.scale[a_idx] = act.scale.data.squeeze()
            vec_layer.shift[a_idx] = act.shift.data.squeeze()


def test_kan_vectorization():
    configs = [(2, 3), (3, 4), (1, 1), (5, 2)]
    batch_sizes = [1, 8, 64, 256]

    for in_f, out_f in configs:
        torch.manual_seed(42)
        kan_loop = KANLayer_loop(in_f, out_f)
        kan_vec  = KANLayer(in_f, out_f)
        copy_weights(kan_loop, kan_vec)

        for B in batch_sizes:
            torch.manual_seed(7 + B)
            x = torch.randn(B, in_f)
            y_loop = kan_loop(x)
            y_vec  = kan_vec(x)

            assert y_loop.shape == y_vec.shape == (B, out_f), \
                f"Shape mismatch: {y_loop.shape} vs {y_vec.shape}"
            assert torch.allclose(y_loop, y_vec, atol=1e-6), \
                f"Output mismatch for ({in_f}->{out_f}, B={B}): " \
                f"max_diff={( y_loop - y_vec).abs().max():.2e}"

            print(f"  ({in_f}->{out_f}, B={B:3d})  OK  " \
                  f"max_diff={(y_loop - y_vec).abs().max():.2e}")

    print("\nAll tests passed.")


if __name__ == "__main__":
    test_kan_vectorization()
