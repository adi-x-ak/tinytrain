import torch
from torch.nn import functional as F

from tinytrain.model import Attention, ModelConfig, Transformer


def test_attention_matches_pytorch_reference_outputs_and_gradients():
    torch.manual_seed(1)
    attention = Attention(ModelConfig(width=16, heads=4, context_length=8)).double()
    x = torch.randn(2, 5, 16, dtype=torch.float64, requires_grad=True)
    actual = attention(x)
    q, k, v = attention.qkv(x).chunk(3, dim=-1)
    q, k, v = [t.view(2, 5, 4, 4).transpose(1, 2) for t in (q, k, v)]
    reference = attention.output(
        F.scaled_dot_product_attention(q, k, v, is_causal=True)
        .transpose(1, 2)
        .contiguous()
        .view(2, 5, 16)
    )
    torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-12)
    parameters = (x, *attention.parameters())
    actual_gradients = torch.autograd.grad(actual.square().sum(), parameters, retain_graph=True)
    reference_gradients = torch.autograd.grad(reference.square().sum(), parameters)
    for actual_gradient, reference_gradient in zip(
        actual_gradients, reference_gradients, strict=True
    ):
        torch.testing.assert_close(actual_gradient, reference_gradient, rtol=1e-9, atol=1e-11)


def test_future_tokens_cannot_change_earlier_predictions():
    model = Transformer(ModelConfig(width=16, context_length=8))
    tokens = torch.arange(8)[None, :]
    changed = tokens.clone()
    changed[:, 4:] = 200
    torch.testing.assert_close(model(tokens)[:, :4], model(changed)[:, :4], rtol=0, atol=0)
