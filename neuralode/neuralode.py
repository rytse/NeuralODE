import torch
import torch.nn as nn

from neuralode import odesol


class OdeintAdjointDiff(torch.autograd.Function):
    """
    Auto-differentiable torch function that behaves like odeint and can be
    differentiated by solving the associated adjoint system, also known as the
    costate equation.
    """

    @staticmethod
    def forward(ctx, f, odeint_fn, z0, ts, *params):
        """
        Solve the IVP

            Given
                dz/dt = f(z, t)
                z(0) = z0
            Find
                z(tf) = ?

        using the specified ODE solver.

        Args:
            ctx:        torch context
            f:          ODE dynamics as a nn.Module
            z0:         initial condition z(ts[0])
            ts:         time values on which to solve for z(t)
            params:     parameters of the dynamics module to save for backprop
            odeint_fn:  base ODE integrator function

        Returns:
            z(t_i) for each t_i in t_eval
        """

        z_traj = odeint_fn(f, z0, ts)

        ctx.save_for_backward(z_traj, ts, *params)
        ctx.f = f
        ctx.odeint_fn = odeint_fn

        return z_traj

    @staticmethod
    def backward(ctx, dLdz_traj):
        """
        Calculate the backprop gradient of an ODE solution with respect to the
        inputs, including the model parameters.

        Args:
            ctx:    torch context
            dLdz:   gradient with respect to the outputs

        Returns:
            Gradient with respect to the inputs dLdp
        """

        # Load values from context
        z_traj, ts, *params = ctx.saved_tensors
        params = tuple(params)
        f = ctx.f
        odeint_fn = ctx.odeint_fn

        # Save relevant dimension
        batch_size, z_dim, n_timesteps = z_traj.shape
        param_dim = sum([param.flatten().shape[0] for param in params])

        # Desired output gradients
        dLdz = torch.zeros(batch_size, z_dim)  # at z(t0)
        dLdt = torch.zeros(batch_size, n_timesteps)  # at z(t0)
        dLdp = torch.zeros(batch_size, param_dim)  # integrated over all t

        # Define augmented costate equation dynamics for use in back-solving
        # for the desired gradients
        def aug_dynamics(aug, t):
            """
            Dynamics of the augmented system containing

                z(t)            state
                a(t)            costate
                dLdt            time derivative contributions
                dLdp            parameter gradient contributions

            Set up to be integrated by odeint_fn

            Args:
                aug augmented system state [z(t), a(t), a(t) df/dp, a(t) df/dt]
                t   time at which to evaluate augmented dynamcis

            Returns:
                Dynamics of the augmented system (daug/dt)
            """

            z = aug[:, :z_dim]
            a = aug[:, z_dim : z_dim * 2]

            # Calculate f(z, t) and its derivatives
            with torch.enable_grad():
                t_ = t.detach()
                t = t_.requires_grad_(True)
                z = z.detach().requires_grad_(True)

                dzdt = f(z, t)

                # Pytorch bug #39784 workaround
                _t = torch.as_strided(t, (), ())
                _z = torch.as_strided(z, (), ())
                _params = tuple(torch.as_strided(p, (), ()) for p in params)

                dfdz, dfdt_singleton, *dfdp_list = torch.autograd.grad(
                    (dzdt,),
                    (z, t) + tuple(f.parameters()),
                    grad_outputs=a,
                    allow_unused=True,
                    retain_graph=True,
                )

            # Flatten dfdp for integration
            dfdp = torch.cat([rep.flatten() for rep in dfdp_list]).unsqueeze(0)
            dfdt = dfdt_singleton.unsqueeze(0)
            if dfdt.dim() < 2:
                dfdt = dfdt.unsqueeze(0)

            dfdp = dfdp.repeat((batch_size, 1))
            dfdt = dfdt.repeat((batch_size, 1))

            # Calculate adjoint and augmented dynamics
            # Don't forget we already multiplied by a in autograd.grad
            dadt = -dfdz
            dadfdt = -dfdt
            dadfdp = -dfdp

            return torch.cat((dzdt, dadt, dadfdt, dadfdp), dim=1)

        # Solve the augmented costate equations backwards in time, running a
        # new odeint solve for each desired output timestep (this is a
        # different time resolution then the odeint solver resolution!)
        with torch.no_grad():
            for ti in range(1, len(ts)):
                # Collect terms for solving the costate equation
                z_f = z_traj[:, :, ti]
                a_f = dLdz_traj[:, :, ti]
                dzdt_f = f(z_f, ts[ti])
                dLdt_f = torch.sum(a_f * dzdt_f, dim=1).unsqueeze(1)
                dLdp_f = torch.zeros(batch_size, param_dim)

                # Solve costate equations with augmented terms for this time step
                aug_f = torch.cat((z_f, a_f, dLdt_f, dLdp_f), dim=1)
                aug = odeint_fn(aug_dynamics, aug_f, (ts[ti], ts[ti - 1]))

                # Add contribution to desired gradient outputs, accumulating dLdp
                # and saving dLdz and dLdt until we get back to dLdz(0), dLdt(0)
                dLdz = aug[:, z_dim : z_dim * 2, -1]  # a(t), updates until last step
                dLdt[:, ti] = aug[:, z_dim * 2 : z_dim * 2 + 1, -1].flatten()
                dLdp += aug[:, z_dim * 2 + 1 :, -1]

        # Unflatten dLdp for final result
        flat_idx = 0
        dLdp_unflattened = []
        for param in params:
            dLdp_unflattened.append(
                dLdp[:, flat_idx : flat_idx + param.flatten().shape[0]].view(
                    (batch_size,) + tuple(param.shape)
                )
            )
            flat_idx += param.flatten().shape[0]

        return None, None, dLdz, dLdt, *dLdp_unflattened


class NeuralODE(nn.Module):
    def __init__(self, f, odesol=odesol.odesol_euler):
        """
        Neural ODE

        Ricky T. Q. Chen, Yulia Rubanova, Jesse Bettencourt, David K. Duvenaud
        Neural Ordinary Differential Equations

        Args:
            f:      ODE dynamics, dz/dt = f(t, x) as a nn.Module
            odesol: ODE solver
        """
        super(NeuralODE, self).__init__()

        self.f = f
        self.odesol = odesol

    def __flat_cat_params(self):
        """
        Get the parameters of the neural ODE dynamics module as a single 1D
        tensor, flattening each (n, m) layer parameter into a (n * m) tensor,
        and concatenating the parameters of each layer into one big tensor.
        """
        params = [*self.f.parameters()]  # unpack generator
        flat_params = [param.flatten() for param in params]
        flat_cat_params = torch.cat(flat_params)
        return flat_cat_params

    def forward(self, z0, t):
        """
        Compute the forward pass of the neural ODE specified by the following
        nn-parameterized IVP:

            dz/dt = f(t, z)
            z(0) = z0

        Args:
            z0: initial condition, z(0) = z0
            t:  time grid to evaluate z on

        Returns:
            Solution trajectory z(t)
        """
        return OdeintAdjointDiff.apply(
            self.f,
            self.odesol,
            z0,
            t,
            *tuple(self.f.parameters()),  # self.__flat_cat_params()
        )
