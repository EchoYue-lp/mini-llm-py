"""Foundation F05: computation graphs, gradients, VJPs, and freezing.

Exercises after running this module:
1. Call ``backward`` twice without clearing ``x.grad`` and observe accumulation.
2. Insert ``detach`` into the residual example and inspect the new gradient.
3. Unfreeze the Linear layer and confirm that ``weight.grad`` is populated.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def polynomial_gradient(value: float) -> tuple[float, float]:
    """Differentiate y=x^2+2x+1 at one scalar value."""
    x = torch.tensor(value, dtype=torch.float32, requires_grad=True)
    y = x.square() + 2 * x + 1
    y.backward()
    return y.item(), x.grad.item()


def vector_jacobian_product() -> torch.Tensor:
    """Return v^T J for a two-output function without building full J."""
    x = torch.tensor([1.0, 2.0], requires_grad=True)
    output = torch.stack((x[0] * x[1], x[0] + x[1]))
    vector = torch.tensor([2.0, 3.0])
    return torch.autograd.grad(output, x, grad_outputs=vector)[0]


def residual_gradient(values: torch.Tensor) -> torch.Tensor:
    """Differentiate sum(x + x^2), exposing the identity gradient term."""
    x = values.detach().clone().requires_grad_(True)
    output = (x + x.square()).sum()
    output.backward()
    return x.grad


def frozen_linear_input_gradient() -> tuple[torch.Tensor, torch.Tensor | None]:
    """Show that a frozen parameter can still pass gradients to its input."""
    torch.manual_seed(0)
    layer = nn.Linear(3, 2)
    layer.requires_grad_(False)
    inputs = torch.randn(4, 3, requires_grad=True)
    loss = layer(inputs).square().mean()
    loss.backward()
    return inputs.grad, layer.weight.grad


def run_demo() -> None:
    output, gradient = polynomial_gradient(3.0)
    assert output == 16.0
    assert gradient == 8.0

    vjp = vector_jacobian_product()
    assert torch.allclose(vjp, torch.tensor([7.0, 5.0]))

    values = torch.tensor([-1.0, 0.0, 2.0])
    gradient_with_residual = residual_gradient(values)
    assert torch.allclose(gradient_with_residual, 1 + 2 * values)

    input_gradient, weight_gradient = frozen_linear_input_gradient()
    assert input_gradient is not None and torch.isfinite(input_gradient).all()
    assert weight_gradient is None

    print("y=x^2+2x+1 at x=3:", output, "gradient:", gradient)
    print("vector-Jacobian product:", vjp.tolist())
    print("residual gradient 1 + 2x:", gradient_with_residual.tolist())
    print("frozen weight grad:", weight_gradient)
    print("input grad norm through frozen layer:", input_gradient.norm().item())


if __name__ == "__main__":
    run_demo()
