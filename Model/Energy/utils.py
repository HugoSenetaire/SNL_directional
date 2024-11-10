import torch


def _eval_poly(y, coef):
    coef = list(coef)
    result = coef.pop()
    while coef:
        result = coef.pop() + y * result
    return result


def get_polar_from_cartesian(
    x,
):
    """
    Convert cartesian coordinates to polar coordinates.
    """

    if x is not None:
        if len(x.shape) > 2 and x.shape[-1] > 3:
            raise NotImplementedError("Only 2D data is supporte if giving coordinates")
        if x.shape[-1] == 2:
            theta = torch.atan2(x[..., 1], x[..., 0])
            return theta
        elif x.shape[-1] == 3:
            r = torch.norm(x, dim=-1)
            theta = torch.acos(x[..., 2] / r)
            r_2d = torch.norm(torch.stack([x[..., 0], x[..., 1]], dim=-1), dim=-1)
            phi = torch.sign(x[..., 1]) * torch.acos(x[..., 0] / r_2d)
            return torch.stack([phi, theta], dim=-1)


def get_cartesian_from_polar(
    theta,
):
    """
    Convert polar coordinates to cartesian coordinates.
    """
    if theta is not None:
        # if ambiant_dim > 3:
        if theta.shape[-1] == 1 or len(theta.shape) == 1:
            x = torch.stack(
                [torch.cos(theta.flatten()), torch.sin(theta.flatten())], dim=-1
            )
        elif theta.shape[-1] == 2:
            x = torch.stack(
                [
                    torch.sin(theta[..., 0]) * torch.cos(theta[..., 1]),
                    torch.sin(theta[..., 0]) * torch.sin(theta[..., 1]),
                    torch.cos(theta[..., 0]),
                ],
                dim=-1,
            )
        else:
            raise NotImplementedError(
                "Only 2D and 3D data is supported if giving coordinates"
            )
        return x


_I0_COEF_SMALL = [
    1.0,
    3.5156229,
    3.0899424,
    1.2067492,
    0.2659732,
    0.360768e-1,
    0.45813e-2,
]
_I0_COEF_LARGE = [
    0.39894228,
    0.1328592e-1,
    0.225319e-2,
    -0.157565e-2,
    0.916281e-2,
    -0.2057706e-1,
    0.2635537e-1,
    -0.1647633e-1,
    0.392377e-2,
]
_I1_COEF_SMALL = [
    0.5,
    0.87890594,
    0.51498869,
    0.15084934,
    0.2658733e-1,
    0.301532e-2,
    0.32411e-3,
]
_I1_COEF_LARGE = [
    0.39894228,
    -0.3988024e-1,
    -0.362018e-2,
    0.163801e-2,
    -0.1031555e-1,
    0.2282967e-1,
    -0.2895312e-1,
    0.1787654e-1,
    -0.420059e-2,
]

_COEF_SMALL = [_I0_COEF_SMALL, _I1_COEF_SMALL]
_COEF_LARGE = [_I0_COEF_LARGE, _I1_COEF_LARGE]


def _log_modified_bessel_fn(x, order=0):
    """
    Returns ``log(I_order(x))`` for ``x > 0``,
    where `order` is either 0 or 1.
    """
    assert order == 0 or order == 1

    # compute small solution
    y = x / 3.75
    y = y * y
    small = _eval_poly(y, _COEF_SMALL[order])
    if order == 1:
        small = x.abs() * small
    small = small.log()

    # compute large solution
    y = 3.75 / x
    large = x - 0.5 * x.log() + _eval_poly(y, _COEF_LARGE[order]).log()

    result = torch.where(x < 3.75, small, large)
    return result
