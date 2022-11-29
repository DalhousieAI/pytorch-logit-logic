"""
Probabilistic Boolean logic activation functions.
"""

from __future__ import division

import functools

import torch
from torch import nn


def maxd(z, dim):
    return torch.max(z, dim=dim).values


def mind(z, dim):
    return torch.min(z, dim=dim).values


def logistic_and(z, dim, eps=1e-7):
    p = torch.sigmoid(z).prod(dim=dim)
    return torch.log(torch.clamp(p, eps)) - torch.log(torch.clamp(1 - p, eps))


def logistic_or(z, dim, eps=1e-7):
    p_inv = torch.sigmoid(-z).prod(dim=dim)
    return torch.log(torch.clamp(1 - p_inv, eps)) - torch.log(torch.clamp(p_inv, eps))


def logistic_xnor_2d(z, dim, eps=1e-7):
    pz = torch.sigmoid(z)
    p = pz.prod(dim=dim) + (1 - pz).prod(dim=dim)
    return torch.log(torch.clamp(p, eps)) - torch.log(torch.clamp(1 - p, eps))


logistic_parity_2d = logistic_xnor_2d


def logistic_parity_3d(z, dim, eps=1e-7):
    # Probability that the number of active arguments is even
    p1, p2, p3 = torch.sigmoid(z).unbind(dim)
    p = (
        (1 - p1) * (1 - p2) * (1 - p3)  # 0 active
        + p1 * p2 * (1 - p3)  # 2 active: (1, 2)
        + p1 * (1 - p2) * p3  # 2 active: (1, 3)
        + (1 - p1) * p2 * p3  # 2 active: (2, 3)
    )
    return torch.log(torch.clamp(p, eps)) - torch.log(torch.clamp(1 - p, eps))


def logistic_parity_4d(z, dim, eps=1e-7):
    p1, p2, p3, p4 = torch.sigmoid(z).unbind(dim)
    # Probability that the number of active arguments is even
    p = (
        (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4)  # 0 active
        + p1 * p2 * (1 - p3) * (1 - p4)  # 2 active: (1, 2)
        + p1 * (1 - p2) * p3 * (1 - p4)  # 2 active: (1, 3)
        + (1 - p1) * p2 * p3 * (1 - p4)  # 2 active: (2, 3)
        + p1 * (1 - p2) * (1 - p3) * p4  # 2 active: (1, 4)
        + (1 - p1) * p2 * (1 - p3) * p4  # 2 active: (2, 4)
        + (1 - p1) * (1 - p2) * p3 * p4  # 2 active: (3, 4)
        + p1 * p2 * p3 * p4  # 4 active: (1, 2, 3, 4)
    )
    return torch.log(torch.clamp(p, eps)) - torch.log(torch.clamp(1 - p, eps))


def logistic_parity_5d(z, dim, eps=1e-7):
    p1, p2, p3, p4, p5 = torch.sigmoid(z).unbind(dim)
    # Probability that the number of active arguments is even
    p = (
        (1 - p1) * (1 - p2) * (1 - p3) * (1 - p4) * (1 - p5)  # 0 active
        + p1 * p2 * (1 - p3) * (1 - p4) * (1 - p5)  # 2 active: (1, 2)
        + p1 * (1 - p2) * p3 * (1 - p4) * (1 - p5)  # 2 active: (1, 3)
        + (1 - p1) * p2 * p3 * (1 - p4) * (1 - p5)  # 2 active: (2, 3)
        + p1 * (1 - p2) * (1 - p3) * p4 * (1 - p5)  # 2 active: (1, 4)
        + (1 - p1) * p2 * (1 - p3) * p4 * (1 - p5)  # 2 active: (2, 4)
        + (1 - p1) * (1 - p2) * p3 * p4 * (1 - p5)  # 2 active: (3, 4)
        + p1 * (1 - p2) * (1 - p3) * (1 - p4) * p5  # 2 active: (1, 5)
        + (1 - p1) * p2 * (1 - p3) * (1 - p4) * p5  # 2 active: (2, 5)
        + (1 - p1) * (1 - p2) * p3 * (1 - p4) * p5  # 2 active: (3, 5)
        + (1 - p1) * (1 - p2) * (1 - p3) * p4 * p5  # 2 active: (4, 5)
        + p1 * p2 * p3 * p4 * (1 - p5)  # 4 active: (1, 2, 3, 4)
        + p1 * p2 * p3 * (1 - p4) * p5  # 4 active: (1, 2, 3, 5)
        + p1 * p2 * (1 - p3) * p4 * p5  # 4 active; (1, 2, 4, 5)
        + p1 * (1 - p2) * p3 * p4 * p5  # 4 active: (1, 3, 4, 5)
        + (1 - p1) * p2 * p3 * p4 * p5  # 4 active; (2, 3, 4, 5)
    )
    return torch.log(torch.clamp(p, eps)) - torch.log(torch.clamp(1 - p, eps))


def logistic_parity(z, dim, **kwargs):
    if z.shape[dim] == 2:
        return logistic_parity_2d(z, dim=dim, **kwargs)
    elif z.shape[dim] == 3:
        return logistic_parity_3d(z, dim=dim, **kwargs)
    elif z.shape[dim] == 4:
        return logistic_parity_4d(z, dim=dim, **kwargs)
    elif z.shape[dim] == 5:
        return logistic_parity_5d(z, dim=dim, **kwargs)
    else:
        raise ValueError("Unsupported k value {}".format(z.shape[dim]))


def logistic_and_normalized(z, dim, **kwargs):
    # mean and stdev values determined empirically, average over 1.2B sample arguments
    if z.shape[dim] == 2:
        mu = -1.29895
        sig = 0.94834
    elif z.shape[dim] == 3:
        mu = -2.27741
        sig = 1.00907
    elif z.shape[dim] == 4:
        mu = -3.1575
        sig = 1.09799
    elif z.shape[dim] == 5:
        mu = -3.9979
        sig = 1.19478
    else:
        raise ValueError("Unsupported k value {}".format(z.shape[dim]))
    return logistic_and(z, dim, **kwargs).sub_(mu).div_(sig)


def logistic_or_normalized(z, dim, **kwargs):
    # mean and stdev values determined empirically, average over 1.2B sample arguments
    if z.shape[dim] == 2:
        mu = 1.29895
        sig = 0.94834
    elif z.shape[dim] == 3:
        mu = 2.27741
        sig = 1.00907
    elif z.shape[dim] == 4:
        mu = 3.1575
        sig = 1.09799
    elif z.shape[dim] == 5:
        mu = 3.9979
        sig = 1.19478
    else:
        raise ValueError("Unsupported k value {}".format(z.shape[dim]))
    return logistic_or(z, dim, **kwargs).sub_(mu).div_(sig)


def logistic_parity_normalized(z, dim, **kwargs):
    # stdev values determined empirically, average over 1.2B sample arguments
    if z.shape[dim] == 2:
        return logistic_parity_2d(z, dim=dim, **kwargs).div_(0.36641)
    elif z.shape[dim] == 3:
        return logistic_parity_3d(z, dim=dim, **kwargs).div_(0.14725)
    elif z.shape[dim] == 4:
        return logistic_parity_4d(z, dim=dim, **kwargs).div_(0.06062)
    elif z.shape[dim] == 5:
        return logistic_parity_5d(z, dim=dim, **kwargs).div_(0.025145)
    else:
        raise ValueError("Unsupported k value {}".format(z.shape[dim]))


def logistic_and_approx_2d(z, dim):
    return torch.where(
        (z < 0).all(dim=dim),
        z.sum(dim=dim),
        torch.min(z, dim=dim).values,
    ).type_as(z)


def logistic_or_approx_2d(z, dim):
    return torch.where(
        (z > 0).all(dim=dim),
        z.sum(dim=dim),
        torch.max(z, dim=dim).values,
    ).type_as(z)


def logistic_xnor_approx_2d(z, dim):
    with torch.no_grad():
        s = torch.sign(torch.prod(z, dim=dim))
    return (s * torch.min(z.abs(), dim=dim).values).type_as(z)


def logistic_and_approx_2d_normalized_old(z, dim):
    # divide by math.sqrt((1 + 2.5 * np.pi) / (2 * math.pi))
    return logistic_and_approx_2d(z, dim).mul_(0.8424043984960415)


def logistic_or_approx_2d_normalized_old(z, dim):
    # divide by math.sqrt((1 + 2.5 * np.pi) / (2 * math.pi))
    return logistic_or_approx_2d(z, dim).mul_(0.8424043984960415)


def logistic_and_approx_2d_normalized2(z, dim):
    # Subtract mean of -1/math.sqrt(2*math.pi)-1/(2*math.sqrt(math.pi))
    # Divide by stdev of math.sqrt(5/4-1/(math.sqrt(2)*math.pi)-1/(4*math.pi))
    return (
        logistic_and_approx_2d(z, dim).add_(0.6810370721753108).div_(0.9722877400310959)
    )


def logistic_or_approx_2d_normalized2(z, dim):
    # Subtract mean of 1/math.sqrt(2*math.pi)+1/(2*math.sqrt(math.pi))
    # Divide by stdev of math.sqrt(5/4-1/(math.sqrt(2)*math.pi)-1/(4*math.pi))
    return (
        logistic_or_approx_2d(z, dim).sub_(0.6810370721753108).div_(0.9722877400310959)
    )


def logistic_xnor_approx_2d_normalized(z, dim):
    # divide by math.sqrt(1 - 2 / math.pi)
    return logistic_xnor_approx_2d(z, dim).mul_(1.658896739970306)


logistic_xnor_approx_2d_normalized2 = logistic_xnor_approx_2d_normalized


def logistic_and_approx(z, dim):
    out_neg = -nn.functional.relu(-z).sum(dim=dim)
    return torch.where(
        out_neg < 0,
        out_neg,
        torch.min(z, dim=dim).values,
    ).type_as(z)


def logistic_or_approx(z, dim):
    out_pos = nn.functional.relu(z).sum(dim=dim)
    return torch.where(
        out_pos > 0,
        out_pos,
        torch.max(z, dim=dim).values,
    ).type_as(z)


def logistic_parity_approx(z, dim):
    with torch.no_grad():
        s = torch.pow(torch.tensor(-1), z.shape[dim]) * torch.sign(
            torch.prod(z, dim=dim)
        )
    return (s * torch.min(z.abs(), dim=dim).values).type_as(z)


def logistic_and_approx_normalized(z, dim):
    # mu and sigma values determined empirically, average over 1.2B sample arguments
    if z.shape[dim] == 2:
        # Values determined analytically
        mu = -0.6810370721753108  # -1/sqrt(2*pi)-1/(2*sqrt(pi))
        sig = 0.9722877400310959  # sqrt(5/4-1/(sqrt(2)*pi)-1/(4*pi))
    elif z.shape[dim] == 3:
        mu = -1.15497
        sig = 1.0701
    elif z.shape[dim] == 4:
        mu = -1.5794
        sig = 1.1929
    elif z.shape[dim] == 5:
        mu = -1.988
        sig = 1.3167
    elif z.shape[dim] == 8:
        # average over 600M samples
        mu = -3.191
        sig = 1.6524
    elif z.shape[dim] == 12:
        # average over 300M samples
        mu = -4.7873
        sig = 2.0225
    elif z.shape[dim] == 16:
        # average over 300M samples
        mu = -6.383
        sig = 2.335
    elif z.shape[dim] == 24:
        # average over 200M samples
        mu = -9.574
        sig = 2.860
    elif z.shape[dim] == 32:
        # average over 200M samples
        mu = -12.766
        sig = 3.302
    else:
        raise ValueError("Unsupported k value {}".format(z.shape[dim]))
    return logistic_and_approx(z, dim).sub_(mu).div_(sig)


def logistic_or_approx_normalized(z, dim):
    # mu and sigma values determined empirically, average over 1.2B sample arguments
    if z.shape[dim] == 2:
        # Values determined analytically
        mu = 0.6810370721753108  # 1/sqrt(2*pi)+1/(2*sqrt(pi))
        sig = 0.9722877400310959  # sqrt(5/4-1/(sqrt(2)*pi)-1/(4*pi))
    elif z.shape[dim] == 3:
        mu = 1.15497
        sig = 1.0701
    elif z.shape[dim] == 4:
        mu = 1.5794
        sig = 1.1929
    elif z.shape[dim] == 5:
        mu = 1.988
        sig = 1.3167
    elif z.shape[dim] == 8:
        # average over 600M samples
        mu = 3.191
        sig = 1.6524
    elif z.shape[dim] == 12:
        # average over 300M samples
        mu = 4.7873
        sig = 2.0225
    elif z.shape[dim] == 16:
        # average over 300M samples
        mu = 6.383
        sig = 2.335
    elif z.shape[dim] == 24:
        # average over 200M samples
        mu = 9.574
        sig = 2.860
    elif z.shape[dim] == 32:
        # average over 200M samples
        mu = 12.766
        sig = 3.302
    else:
        raise ValueError("Unsupported k value {}".format(z.shape[dim]))
    return logistic_or_approx(z, dim).sub_(mu).div_(sig)


def logistic_parity_approx_normalized(z, dim):
    # mu and sigma values determined empirically, average over 1.2B sample arguments
    if z.shape[dim] == 2:
        # Value determined analytically
        sig = 0.6028102749890869  # sqrt(1 - 2 / pi)
    elif z.shape[dim] == 3:
        sig = 0.4391
    elif z.shape[dim] == 4:
        sig = 0.3474
    elif z.shape[dim] == 5:
        sig = 0.2882
    elif z.shape[dim] == 8:
        # average over 600M samples
        sig = 0.1918
    elif z.shape[dim] == 12:
        # average over 300M samples
        sig = 0.1332
    elif z.shape[dim] == 16:
        # average over 300M samples
        sig = 0.1022
    elif z.shape[dim] == 24:
        # average over 200M samples
        sig = 0.06982
    elif z.shape[dim] == 32:
        # average over 200M samples
        sig = 0.05305
    else:
        raise ValueError("Unsupported k value {}".format(z.shape[dim]))
    return logistic_parity_approx(z, dim).div_(sig)


def unroll_k(x, k, d):
    if x.shape[d] % k != 0:
        raise ValueError(
            "Argument {} has shape {}. Dimension {} is {}, which is not"
            " divisible by {}.".format(x, x.shape, d, x.shape[d], k)
        )
    shp = list(x.shape)
    d_a = d % len(shp)
    shp = shp[:d_a] + [x.shape[d_a] // k] + [k] + shp[d_a + 1 :]
    x = x.view(*shp)
    d_new = d if d < 0 else d + 1
    return x, d_new


class GLU(nn.GLU):
    k = 2

    @property
    def divisor(self):
        return self.k

    @property
    def feature_factor(self):
        return 1 / self.k


class HOActfun(nn.Module):
    def __init__(self, k=2, dim=1):
        super(HOActfun, self).__init__()
        self.k = k
        self.dim = dim

    @property
    def divisor(self):
        return self.k

    @property
    def feature_factor(self):
        return 1 / self.k


class MaxOut(HOActfun):
    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return torch.max(x, dim=d_new).values


class MinOut(HOActfun):
    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return torch.min(x, dim=d_new).values


class SignedGeomeanFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, dim, keepdim, clamp_grad):
        # Save inputs
        ctx.save_for_backward(input)
        ctx.dim = dim + input.ndim if dim < 0 else dim
        ctx.keepdim = keepdim
        ctx.clamp_grad = clamp_grad
        # Compute forward pass
        prods = input.prod(dim=dim, keepdim=keepdim)
        signs = prods.sign()
        output = signs * prods.abs().sqrt()
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        if not ctx.keepdim:
            grad_output = grad_output.unsqueeze(ctx.dim)

        # Re-compute forward pass
        prods = input.prod(dim=ctx.dim, keepdim=True)
        signs = prods.sign()
        output = signs * prods.abs().sqrt()

        grad_inner = 0.5 * output / input
        # Remove singularities
        grad_inner[input.abs() == 0] = 0
        # Clamp large values
        if ctx.clamp_grad is not None:
            grad_inner = torch.clamp(grad_inner, -ctx.clamp_grad, ctx.clamp_grad)
        # dy/dx = dy/dz * dz/dx
        grad_input = grad_output * grad_inner

        # Need to return None for each non-tensor input to forward
        return grad_input, None, None, None


def signed_geomean(x, dim=1, keepdim=False, clamp_grad=None):
    return SignedGeomeanFunc.apply(x, dim, keepdim, clamp_grad)


class SignedGeomean(HOActfun):
    def __init__(self, clamp_grad=None, *args, **kwargs):
        super(SignedGeomean, self).__init__(*args, **kwargs)
        self.clamp_grad = clamp_grad

    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return signed_geomean(x, d_new, clamp_grad=self.clamp_grad)


class IL_AND(HOActfun):
    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return logistic_and(x, d_new)


class IL_OR(HOActfun):
    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return logistic_or(x, d_new)


class IL_Parity(HOActfun):
    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return logistic_parity(x, d_new)


IL_XNOR = IL_Parity


class NIL_AND(HOActfun):
    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return logistic_and_normalized(x, d_new)


class NIL_OR(HOActfun):
    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return logistic_or_normalized(x, d_new)


class NIL_Parity(HOActfun):
    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return logistic_parity_normalized(x, d_new)


NIL_XNOR = NIL_Parity


class AIL_AND(HOActfun):
    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return logistic_and_approx(x, d_new)


class AIL_OR(HOActfun):
    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return logistic_or_approx(x, d_new)


class AIL_Parity(HOActfun):
    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return logistic_parity_approx(x, d_new)


AIL_XNOR = AIL_Parity


class NAIL_AND(HOActfun):
    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return logistic_and_approx_normalized(x, d_new)


class NAIL_OR(HOActfun):
    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return logistic_or_approx_normalized(x, d_new)


class NAIL_Parity(HOActfun):
    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return logistic_parity_approx_normalized(x, d_new)


NAIL_XNOR = NAIL_Parity


class MultiActfunDuplicate(HOActfun):
    def __init__(self, actfuns, **kwargs):
        super(MultiActfunDuplicate, self).__init__(**kwargs)
        self.actfuns = actfuns

    def forward(self, x):
        x, d_new = unroll_k(x, self.k, self.dim)
        return torch.cat([f(x, d_new) for f in self.actfuns], dim=self.dim)

    @property
    def feature_factor(self):
        return len(self.actfuns) / self.k


class MultiActfunPartition(HOActfun):
    def __init__(self, actfuns, **kwargs):
        super(MultiActfunPartition, self).__init__(**kwargs)
        self.actfuns = actfuns

    def forward(self, x):
        dim = self.dim % x.dim()  # Handle negative indexing
        x, d_new = unroll_k(x, self.k, self.dim)
        xs = torch.split(x, x.shape[dim] // len(self.actfuns), dim=dim)
        return torch.cat(
            [f(xi, d_new) for f, xi in zip(self.actfuns, xs)], dim=self.dim
        )

    @property
    def divisor(self):
        return len(self.actfuns) * self.k


class max_min_duplicate(MultiActfunDuplicate):
    def __init__(self, **kwargs):
        super(max_min_duplicate, self).__init__([maxd, mind], **kwargs)


class IL_AND_OR_duplicate(MultiActfunDuplicate):
    def __init__(self, **kwargs):
        super(IL_AND_OR_duplicate, self).__init__([logistic_and, logistic_or], **kwargs)


class IL_OR_XNOR_duplicate(MultiActfunDuplicate):
    def __init__(self, **kwargs):
        super(IL_OR_XNOR_duplicate, self).__init__(
            [logistic_or, logistic_parity], **kwargs
        )


class IL_AND_OR_XNOR_duplicate(MultiActfunDuplicate):
    def __init__(self, **kwargs):
        super(IL_AND_OR_XNOR_duplicate, self).__init__(
            [logistic_and, logistic_or, logistic_parity], **kwargs
        )


class NIL_AND_OR_duplicate(MultiActfunDuplicate):
    def __init__(self, **kwargs):
        super(NIL_AND_OR_duplicate, self).__init__(
            [logistic_and_normalized, logistic_or_normalized], **kwargs
        )


class NIL_OR_XNOR_duplicate(MultiActfunDuplicate):
    def __init__(self, **kwargs):
        super(NIL_OR_XNOR_duplicate, self).__init__(
            [logistic_or_normalized, logistic_parity_normalized], **kwargs
        )


class NIL_AND_OR_XNOR_duplicate(MultiActfunDuplicate):
    def __init__(self, **kwargs):
        super(NIL_AND_OR_XNOR_duplicate, self).__init__(
            [
                logistic_and_normalized,
                logistic_or_normalized,
                logistic_parity_normalized,
            ],
            **kwargs,
        )


class AIL_AND_OR_duplicate(MultiActfunDuplicate):
    def __init__(self, **kwargs):
        super(AIL_AND_OR_duplicate, self).__init__(
            [logistic_and_approx, logistic_or_approx], **kwargs
        )


class AIL_OR_XNOR_duplicate(MultiActfunDuplicate):
    def __init__(self, **kwargs):
        super(AIL_OR_XNOR_duplicate, self).__init__(
            [logistic_or_approx, logistic_parity_approx], **kwargs
        )


class AIL_AND_OR_XNOR_duplicate(MultiActfunDuplicate):
    def __init__(self, **kwargs):
        super(AIL_AND_OR_XNOR_duplicate, self).__init__(
            [logistic_and_approx, logistic_or_approx, logistic_parity_approx], **kwargs
        )


class NAIL_AND_OR_duplicate(MultiActfunDuplicate):
    def __init__(self, **kwargs):
        super(NAIL_AND_OR_duplicate, self).__init__(
            [logistic_and_approx_normalized, logistic_or_approx_normalized], **kwargs
        )


class NAIL_OR_XNOR_duplicate(MultiActfunDuplicate):
    def __init__(self, **kwargs):
        super(NAIL_OR_XNOR_duplicate, self).__init__(
            [logistic_or_approx_normalized, logistic_parity_approx_normalized], **kwargs
        )


class NAIL_AND_OR_XNOR_duplicate(MultiActfunDuplicate):
    def __init__(self, **kwargs):
        super(NAIL_AND_OR_XNOR_duplicate, self).__init__(
            [
                logistic_and_approx_normalized,
                logistic_or_approx_normalized,
                logistic_parity_approx_normalized,
            ],
            **kwargs,
        )


class max_min_partition(MultiActfunPartition):
    def __init__(self, **kwargs):
        super(max_min_partition, self).__init__([maxd, mind], **kwargs)


class IL_OR_XNOR_partition(MultiActfunPartition):
    def __init__(self, **kwargs):
        super(IL_OR_XNOR_partition, self).__init__(
            [logistic_or, logistic_parity], **kwargs
        )


class IL_AND_OR_XNOR_partition(MultiActfunPartition):
    def __init__(self, **kwargs):
        super(IL_AND_OR_XNOR_partition, self).__init__(
            [logistic_and, logistic_or, logistic_parity], **kwargs
        )


class NIL_OR_XNOR_partition(MultiActfunPartition):
    def __init__(self, **kwargs):
        super(NIL_OR_XNOR_partition, self).__init__(
            [logistic_or_normalized, logistic_parity_normalized], **kwargs
        )


class NIL_AND_OR_XNOR_partition(MultiActfunPartition):
    def __init__(self, **kwargs):
        super(NIL_AND_OR_XNOR_partition, self).__init__(
            [
                logistic_and_normalized,
                logistic_or_normalized,
                logistic_parity_normalized,
            ],
            **kwargs,
        )


class AIL_OR_XNOR_partition(MultiActfunPartition):
    def __init__(self, **kwargs):
        super(AIL_OR_XNOR_partition, self).__init__(
            [logistic_or_approx, logistic_parity_approx], **kwargs
        )


class AIL_AND_OR_XNOR_partition(MultiActfunPartition):
    def __init__(self, **kwargs):
        super(AIL_AND_OR_XNOR_partition, self).__init__(
            [logistic_and_approx, logistic_or_approx, logistic_parity_approx], **kwargs
        )


class NAIL_OR_XNOR_partition(MultiActfunPartition):
    def __init__(self, **kwargs):
        super(NAIL_OR_XNOR_partition, self).__init__(
            [logistic_or_approx_normalized, logistic_parity_approx_normalized], **kwargs
        )


class NAIL_AND_OR_XNOR_partition(MultiActfunPartition):
    def __init__(self, **kwargs):
        super(NAIL_AND_OR_XNOR_partition, self).__init__(
            [
                logistic_and_approx_normalized,
                logistic_or_approx_normalized,
                logistic_parity_approx_normalized,
            ],
            **kwargs,
        )


_ACT_LAYER_LOGICAL = {
    "max": MaxOut,
    "maxout": MaxOut,
    "signedgeomean": SignedGeomean,
    "signedgeomean_clamp2": functools.partial(SignedGeomean, clamp_grad=2),
    "signedgeomean_clamp10": functools.partial(SignedGeomean, clamp_grad=10),
    "il_and": IL_AND,
    "il_or": IL_OR,
    "il_xnor": IL_XNOR,
    "nil_and": NIL_AND,
    "nil_or": NIL_OR,
    "nil_xnor": NIL_XNOR,
    "ail_and": AIL_AND,
    "ail_or": AIL_OR,
    "ail_xnor": AIL_XNOR,
    "nail_and": NAIL_AND,
    "nail_or": NAIL_OR,
    "nail_xnor": NAIL_XNOR,
    "max_min_dup": max_min_duplicate,
    "il_and_or_dup": IL_AND_OR_duplicate,
    "il_or_xnor_dup": IL_OR_XNOR_duplicate,
    "il_and_or_xnor_dup": IL_AND_OR_XNOR_duplicate,
    "nil_and_or_dup": NIL_AND_OR_duplicate,
    "nil_or_xnor_dup": NIL_OR_XNOR_duplicate,
    "nil_and_or_xnor_dup": NIL_AND_OR_XNOR_duplicate,
    "ail_and_or_dup": AIL_AND_OR_duplicate,
    "ail_or_xnor_dup": AIL_OR_XNOR_duplicate,
    "ail_and_or_xnor_dup": AIL_AND_OR_XNOR_duplicate,
    "nail_and_or_dup": NAIL_AND_OR_duplicate,
    "nail_or_xnor_dup": NAIL_OR_XNOR_duplicate,
    "nail_and_or_xnor_dup": NAIL_AND_OR_XNOR_duplicate,
    "max_min_part": max_min_partition,
    "il_or_xnor_part": IL_OR_XNOR_partition,
    "il_and_or_xnor_part": IL_AND_OR_XNOR_partition,
    "nil_or_xnor_part": NIL_OR_XNOR_partition,
    "nil_and_or_xnor_part": NIL_AND_OR_XNOR_partition,
    "ail_or_xnor_part": AIL_OR_XNOR_partition,
    "ail_and_or_xnor_part": AIL_AND_OR_XNOR_partition,
    "nail_or_xnor_part": NAIL_OR_XNOR_partition,
    "nail_and_or_xnor_part": NAIL_AND_OR_XNOR_partition,
    "max_min_part_fixed": max_min_partition,
    "il_or_xnor_part_fixed": IL_OR_XNOR_partition,
    "il_and_or_xnor_part_fixed": IL_AND_OR_XNOR_partition,
    "nil_or_xnor_part_fixed": NIL_OR_XNOR_partition,
    "nil_and_or_xnor_part_fixed": NIL_AND_OR_XNOR_partition,
    "ail_or_xnor_part_fixed": AIL_OR_XNOR_partition,
    "ail_and_or_xnor_part_fixed": AIL_AND_OR_XNOR_partition,
    "nail_or_xnor_part_fixed": NAIL_OR_XNOR_partition,
    "nail_and_or_xnor_part_fixed": NAIL_AND_OR_XNOR_partition,
}


def actfun_name2factory(name, k=None):
    lower_name = name.lower()
    if lower_name == "relu":
        if k is not None and k != 1:
            raise ValueError("ReLU only supports k=1, not {}".format(k))
        return nn.ReLU
    elif lower_name == "glu":
        if k is not None and k != 2:
            raise ValueError("GLU only supports k=2, not {}".format(k))
        return GLU
    elif name in _ACT_LAYER_LOGICAL:
        if k is not None and k < 2:
            raise ValueError(
                "Higher order activations only support k>=2, not {}".format(k)
            )
        return functools.partial(_ACT_LAYER_LOGICAL[name], k=k or 2)
    elif hasattr(nn, name):
        if k is not None and k != 1:
            raise ValueError("torch.nn activations only support k=1, not {}".format(k))
        return getattr(nn, name)
    else:
        raise ValueError("Unsupported actfun: {}".format(name))
