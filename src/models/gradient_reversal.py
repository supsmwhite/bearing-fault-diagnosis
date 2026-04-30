"""Define the Gradient Reversal Layer used by DANN for adversarial domain adaptation."""

import torch
from torch import nn


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_value):
        ctx.lambda_value = lambda_value
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_value * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self, lambda_value: float = 1.0):
        super().__init__()
        self.lambda_value = lambda_value

    def set_lambda(self, lambda_value: float) -> None:
        self.lambda_value = lambda_value

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_value)
