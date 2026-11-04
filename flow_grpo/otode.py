"""OT-ODE numerics. All public time arguments are normalized, shifted sigma."""
import math

import torch


TEACHER_SOLVERS = ("euler", "heun", "dpm")


def shift_sigma(u, shift):
    return shift * u / (1 + (shift - 1) * u)


def unshift_sigma(sigma, shift):
    return sigma / (shift - (shift - 1) * sigma)


def teacher_sigma_grid(sigma, sigma_next, substeps=8, shift=3.0):
    """Uniform pre-shift subdivisions, mapped back to actual integration time.

    Scalar endpoints produce [substeps+1]; vector endpoints produce
    [substeps+1, batch]. The actual endpoints (including zero) are retained.
    """
    if not isinstance(substeps, int) or isinstance(substeps, bool) or substeps < 1:
        raise ValueError("teacher_substeps must be a positive integer")
    if not math.isfinite(shift) or shift <= 0:
        raise ValueError("flow_shift must be finite and positive")
    sigma = torch.as_tensor(sigma, dtype=torch.float32)
    sigma_next = torch.as_tensor(sigma_next, device=sigma.device, dtype=torch.float32)
    sigma, sigma_next = torch.broadcast_tensors(sigma, sigma_next)
    valid = torch.isfinite(sigma) & torch.isfinite(sigma_next)
    valid &= (sigma <= 1) & (sigma_next >= 0) & (sigma > sigma_next)
    if not bool(valid.all()):
        raise ValueError("Expected 0 <= sigma_next < sigma <= 1")
    u, u_next = unshift_sigma(sigma, shift), unshift_sigma(sigma_next, shift)
    fraction = torch.linspace(0, 1, substeps + 1, device=sigma.device)
    fraction = fraction.reshape(-1, *([1] * sigma.ndim))
    grid = shift_sigma(u.unsqueeze(0) + fraction * (u_next - u).unsqueeze(0), shift)
    grid[0], grid[-1] = sigma, sigma_next
    return grid


def _broadcast_time(t, x):
    return t.reshape(*t.shape, *([1] * (x.ndim - t.ndim))) if t.ndim else t


def _log_snr(sigma):
    return torch.log1p(-sigma) - torch.log(sigma)


@torch.no_grad()
def integrate_teacher(x, sigma, sigma_next, velocity, substeps=8, shift=3.0, solver="heun"):
    """Integrate dx/dsigma=v(x,sigma) from sigma to sigma_next; velocity receives actual sigma.

    euler: 1 call per substep. heun: 2 calls per substep, including the
    endpoint at zero. dpm: DPM-Solver++(2M) on x0 = x - sigma * v, 1 call per
    substep, first order on the first substep and on the step to zero.
    This never touches a stateful DPM scheduler and never retains a teacher graph.
    """
    if solver not in TEACHER_SOLVERS:
        raise ValueError(f"teacher_solver must be one of {TEACHER_SOLVERS}, got {solver!r}")
    grid = teacher_sigma_grid(sigma, sigma_next, substeps, shift).to(x.device)
    state = x.detach().float().clone()
    previous_x0 = previous_sigma = None
    for start, end in zip(grid[:-1], grid[1:]):
        dt = _broadcast_time(end - start, state)
        v_start = velocity(state, start).float()
        if solver == "euler":
            state = state + dt * v_start
        elif solver == "heun":
            proposal = state + dt * v_start
            v_end = velocity(proposal, end).float()
            state = state + (0.5 * dt) * (v_start + v_end)
        else:
            s, s_next = start.double(), end.double()
            x0 = state - _broadcast_time(start, state) * v_start
            # exp(-h) as a ratio so sigma=1 and sigma_next=0 stay finite.
            decay = (1 - s) * s_next / (s * (1 - s_next))
            data = x0
            if previous_x0 is not None:
                s_prev = previous_sigma.double()
                second = (s_prev < 1) & (s_next > 0)
                h = _log_snr(s_next.clamp(min=1e-12)) - _log_snr(s)
                h_prev = _log_snr(s) - _log_snr(s_prev.clamp(max=1 - 1e-12))
                weight = torch.where(second, 0.5 * h / h_prev, torch.zeros_like(h))
                data = x0 + _broadcast_time(weight.float(), state) * (x0 - previous_x0)
            state = (_broadcast_time((s_next / s).float(), state) * state
                     + _broadcast_time(((1 - s_next) * (1 - decay)).float(), state) * data)
            previous_x0, previous_sigma = x0, start
    return state.detach()
