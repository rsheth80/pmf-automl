import torch
import torch.nn as nn
import torch.distributions as dist
import numpy as np
from utils import *

# (1) not sure why dtype is explicitly required in some places to force float32
dtype = torch.float32

class GP(nn.Module):

    def __init__(self, dim, X, y, kernel, variance=1.0, N_max=None):
        super(GP, self).__init__()

        self.dim = torch.tensor([dim], requires_grad=False)
        self.kernel = kernel
        self.variance = torch.nn.Parameter(
                                transform_backward(torch.tensor([variance])))

        if torch.is_tensor(X):
            self.X = X
        else:
            self.X = torch.tensor(X, requires_grad=False, dtype=dtype)

        self.N_max = N_max
        self.N = self.X.size()[0]

        if isinstance(y, Sparse1DTensor):
            self.y = y
            ix = torch.tensor([k for k in y.ix.keys()], dtype=torch.int64)
            self.get_batch = BatchIndices(None, ix, self.N_max)
        else:
            # NOTE: see (1)
            self.y = torch.tensor(y.squeeze(), dtype=dtype,
                                  requires_grad=False)
            self.get_batch = BatchIndices(self.N, None, self.N_max)

    def get_cov(self, ix=None):

        if ix is None:
            ix = torch.arange(0, self.N)

        return torch.potrf(self.kernel(self.X[ix])
                           + torch.eye(ix.numel())
                                *transform_forward(self.variance),
                           upper=False)

    def forward(self, ix=None):

        if ix is None:
            ix = self.get_batch()

        mn = torch.zeros(ix.numel())
        cov = self.get_cov(ix=ix)
        pdf = dist.multivariate_normal.MultivariateNormal(mn, scale_tril=cov)

        return -pdf.log_prob(self.y[ix])

    def posterior(self, Xtest):
        # assumes stationary kernel

        with torch.no_grad():
            if isinstance(self.y, Sparse1DTensor):
                ix = self.get_batch.ix
                Ks = self.kernel(self.X[ix], Xtest)
                L = self.get_cov(ix)
                alpha = torch.trtrs(Ks, L, upper=False)[0]
                fmean = torch.matmul(torch.t(alpha),
                                     torch.trtrs(self.y.v.squeeze(), L,
                                                 upper=False)[0])
            else:
                Ks = self.kernel(self.X, Xtest)
                L = self.get_cov()
                alpha = torch.trtrs(Ks, L, upper=False)[0]
                fmean = torch.matmul(torch.t(alpha),
                                     torch.trtrs(self.y, L, upper=False)[0])
            fvar = transform_forward(self.kernel.variance) - (alpha**2).sum(0)

            return fmean, fvar.reshape((-1,1))

class GPLVM(nn.Module):

    def __init__(self, dim, X, Y, kernel, D_max=None, **kwargs):
        super(GPLVM, self).__init__()

        if torch.is_tensor(X):
            self.X = torch.nn.Parameter(X)
        else:
            # NOTE: see (1)
            self.X = torch.nn.Parameter(torch.tensor(X, dtype=dtype))

        self.GPs = nn.ModuleList([])
        if isinstance(Y, np.ndarray):
            self.D = Y.shape[1]
            for d in range(self.D):
                ix = np.where(np.invert(np.isnan(Y[:,d])))[0]
                y = Sparse1DTensor(Y[ix,d], torch.tensor(ix))
                self.GPs.append(GP(dim, self.X, y, kernel, **kwargs))
        elif isinstance(Y, list):
            # assumes col indexing starts at 0 and is (integer-)continuous
            self.D = int(np.max(Y[2])) + 1
            for d in range(self.D):
                ix = np.where(Y[2]==d)[0]
                y = Sparse1DTensor(Y[0][ix], torch.tensor(Y[1][ix]))
                self.GPs.append(GP(dim, self.X, y, kernel, **kwargs))
        else:
            assert False, 'Bad Y input'

        if D_max is None:
            self.D_max = self.D
        else:
            self.D_max = D_max

        self.dim = dim
        self.kernel = kernel
        for j in range(1, self.D):
            self.GPs[j].variance = self.GPs[0].variance
        self.variance = self.GPs[0].variance

        self.get_batch = BatchIndices(self.D, None, self.D_max)

    def forward(self, ix=None):

        if ix is None:
            ix = self.get_batch()

        lp = torch.tensor([0.])
        for j in ix:
            lp += self.GPs[j]()

        return lp*self.D/self.D_max
