import torch
import torch.nn as nn
import torch.nn.functional as F


def __odesol_euler_step(f, x0, t0, tf, n_step=100):
    x = x0
    t = t0
    dt = (tf - t0) / n_step

    for _ in range(n_step):
        dxdt = f(x, t)
        x += dxdt * dt
        t += dt

    return x


def odesol_euler(f, x0, ts):
    """
    Solve the given IVP using Euler's algorithm. Evaluate on torch tensors,
    using GPU if available.

    Args:
        f:      dx/dt = f(t, x)
        x0:     initial state
        t:      time grid to evaluate x on

    Returns:
        Solution trajectory x(t)
    """

    batch_size, x_dim = x0.shape
    n_timesteps = len(ts)

    x_traj = torch.zeros((batch_size, x_dim, n_timesteps))
    x_traj[:, :, 0] = x0

    for i in range(1, n_timesteps):
        x_traj[:, :, i] = __odesol_euler_step(f, x_traj[:, :, i - 1], ts[i - 1], ts[i])

    return x_traj


def odesol_rk_e(f, x0, t, a, b, c):
    """
    Solve the given IVP using the explicit Runge-Kutta method specified by the
    Butcher tableau with the given coefficients. Evaluate on torch tensors,
    using GPU if available.

    Args:
        f:      dx/dt = f(t, x)
        x0:     initial state
        t:      time grid to evaluate x on
        a:      RK a coefficients ( s x (s-1) )
        b:      RK b coefficients ( s )
        c:      RK c coefficients ( s )

    Returns:
        Solution trajectory x(t)
    """
    x = torch.zeros(len(t), len(x0))
    x[0, :] = x0

    # Integrate over time
    for i in range(1, len(t)):
        h = t[i] - t[i - 1]
        k = torch.zeros(len(b), len(x0))
        k[0, :] = f(x[i - 1, :], t[i - 1])

        # Solve for k[i]
        for j in range(1, len(b)):
            ksum = a[j, 0:j] @ k[0:j, :]
            k[j, :] = f(x[i - 1] + ksum * h, t[i - 1] + c[j] * h)

        x[i, :] = x[i - 1, :] + h * b @ k

    return x


def odesol_dopri(f, x0, t):
    """
    Solve the given IVP using the Dormand-Prince method (i.e. DOPRI, RKDP), a
    popular explicit Runge-Kutta method. Evaluate on torch tensors, using
    GPU if available.

    Args:
        f:      dx/dt = f(t, x)
        x0:     initial state
        t:      time grid to evaluate x on

    Returns:
        Solution trajectory x(t)
    """
    a = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0],
            [1 / 5, 0, 0, 0, 0, 0],
            [3 / 40, 9 / 40, 0, 0, 0, 0],
            [44 / 45, -56 / 15, 32 / 9, 0, 0, 0],
            [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0, 0],
            [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656, 0],
            [35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84],
        ]
    )
    b = torch.tensor(
        [5179 / 57600, 0, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40]
    )
    c = torch.tensor([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1])

    return odesol_rk_e(f, x0, t, a, b, c)
