import torch
import numpy as np

# (1) not sure why dtype is explicitly required in some places to force float32
dtype = torch.float

transform_forward = lambda x: (1+x.exp()).log()
transform_backward = lambda x: (x.exp()-1).log()

class Sparse1DTensor():

    def __init__(self, y, ix):

        # NOTE: see (1)
        self.v = torch.tensor(y, dtype=dtype, requires_grad=False)
        ix_tensor = torch.tensor(ix)
        assert self.v.numel() == ix_tensor.numel(), 'inputs must be same size'
        self.ix = {ix_tensor[i].item(): i for i in torch.arange(self.v.numel())}

    def __getitem__(self, k):

        if not len(k.size()):
            return self.v[self.ix[k.item()]]
        else:
            return torch.tensor([self.v[self.ix[kk]] for kk in k.tolist()])

    def __setitem__(self, k, v):

        if not len(k.size()):
            self.v[self.ix[k.item()]] = v
        else:
            for kk,vv in zip(k.tolist(), v.tolist()):
                self.v[self.ix[kk]] = vv

class BatchIndices():

    def __init__(self, N=None, ix=None, B=None):

        assert (N is not None) or (ix is not None), \
                                    'either N or ix should be provided'
        if (N is not None) and (ix is not None):
            assert N==ix.numel(), 'N must = size of ix'
            self.N = N
            self.ix = ix
        elif N is not None:
            self.N = N
            self.ix = torch.arange(N)
        else:
            self.ix = ix
            self.N = ix.numel()

        if B is None:
            self.B = self.N
        else:
            if B > self.N:
                B = self.N
            self.B = B

        self.perm = torch.randperm(self.N, requires_grad=False)

    def __call__(self, B=None):

        if B is None:
            B = self.B
        else:
            assert B <= self.N, 'Batch size must be <= data size'

        # the no. of random indices to retrieve from current permutation
        m = torch.min(torch.tensor([B, self.perm.numel()]))
        # negative of the no. of elements remaining in current permutation
        # after random indices have been retrieved
        d = self.perm.numel() - B
        # retrieve random indices from current permutation
        ix_batch = self.perm[:m]

        if d <= 0: # current permutation has run out
            # generate new permutation
            self.perm = torch.randperm(self.N)
            if d < 0: # random indices still needed
                # fill remainder of random indices with beginning of new perm
                ix_batch = torch.cat((ix_batch, self.perm[:-d]))
                self.perm = self.perm[-d:]

        else: # current permutation still has entries
            # discard used entries from current permutation
            self.perm = self.perm[m:]

        return self.ix[ix_batch]
