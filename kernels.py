import torch
import torch.nn as nn
from utils import transform_forward, transform_backward

def sqdist(X, Y):

    assert X.size()[1] == Y.size()[1], 'dims do not match'

    return ((X.reshape(X.size()[0], 1, X.size()[1])
             - Y.reshape(1, Y.size()[0], Y.size()[1]))**2).sum(2)

class Constant(nn.Module):

    def __init__(self, variance=1.0):
        super(Constant, self).__init__()

        self.variance = torch.nn.Parameter(
                            transform_backward(torch.tensor([variance])))

    def forward(self, X, X2=None):

        if X2 is None:
            shape = [X.size()[0], X.size()[0]]
        else:
            shape = [X.size()[0], X2.size()[0]]

        return transform_forward(self.variance)*torch.ones(shape[0], shape[1])

class RBF(nn.Module):

    def __init__(self, dim, variance=1.0, lengthscale=None):
        super(RBF, self).__init__()

        self.dim = torch.tensor([dim], requires_grad=False)
        if lengthscale is None:
            self.lengthscale \
                = torch.nn.Parameter(transform_backward(torch.ones(1, dim)))
        else:
            self.lengthscale \
                = torch.nn.Parameter(
                    transform_backward(torch.tensor(lengthscale)))
        self.variance = torch.nn.Parameter(
                            transform_backward(torch.tensor([variance])))

    def forward(self, X, X2=None):

        if X2 is None:
            X2 = X

        l = transform_forward(self.lengthscale)

        return transform_forward(self.variance)*(-0.5*sqdist(X/l, X2/l)).exp()

class Linear(nn.Module):

    def __init__(self, dim, variance=1.0, lengthscale=None):
        super(Linear, self).__init__()

        self.dim = torch.tensor([dim], requires_grad=False)
        if lengthscale is None:
            self.lengthscale \
                = torch.nn.Parameter(transform_backward(torch.ones(1, dim)))
        else:
            self.lengthscale \
                = torch.nn.Parameter(
                    transform_backward(torch.tensor(lengthscale)))
        self.variance = torch.nn.Parameter(
                            transform_backward(torch.tensor([variance])))

    def forward(self, X, X2=None):

        if X2 is None:
            X2 = X

        l = transform_forward(self.lengthscale)

        return transform_forward(self.variance)*torch.mm(X/l, (X2/l).t())

class White(nn.Module):
    # when X != X2, K(X, X2) = 0

    def __init__(self, dim, variance=1.0):
        super(White, self).__init__()

        self.dim = torch.tensor([dim], requires_grad=False)
        self.variance = torch.nn.Parameter(
                            transform_backward(torch.tensor([variance])))

    def forward(self, X, X2=None):

        if X2 is None:
            return torch.eye(X.size()[0])*transform_forward(self.variance)
        else:
            return 0.

class Add(nn.Module):

    def __init__(self, k1, k2):
        super(Add, self).__init__()

        self.k1 = k1
        self.k2 = k2

    @property
    def variance(self):
        return transform_backward(transform_forward(self.k1.variance)
                                  + transform_forward(self.k2.variance))

    def forward(self, X, X2=None):
        return self.k1(X, X2) + self.k2(X, X2)
