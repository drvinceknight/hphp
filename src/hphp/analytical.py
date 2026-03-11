from __future__ import annotations

from collections.abc import Callable, Iterator

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq


def get_x_T_prime(x_T: float, alpha: float, p: float, q: float, *, clip: bool = True) -> float:
    """One-step map x_T -> x_T'.

        x' = (x·α·(1-q) + (1-x)·p) / (1 + x·(α-1))
    """
    x = float(x_T)
    denom = 1.0 + x * (alpha - 1.0)
    x_prime = 0.0 if denom == 0.0 else (x * alpha * (1.0 - q) + (1.0 - x) * p) / denom
    return float(np.clip(x_prime, 0.0, 1.0)) if clip else x_prime


def simulation(x_T: float, alpha: float, p: float, q: float, number_of_iterations: int) -> Iterator[float]:
    """Yield x_T(t) for t = 0 .. number_of_iterations."""
    yield float(x_T)
    for _ in range(int(number_of_iterations)):
        x_T = get_x_T_prime(x_T, alpha, p, q)
        yield float(x_T)


def x_star_equilibrium(alpha: float, p: float, q: float) -> float:
    """Closed-form equilibrium x* in [0, 1]."""
    if p > 0:
        B = 1 + p - alpha + alpha * q
        disc = B**2 + 4 * p * (alpha - 1)
        return float(np.clip((-B + np.sqrt(disc)) / (2 * (alpha - 1)), 0.0, 1.0))
    if alpha * (1 - q) > 1:
        return float(np.clip(1 - (alpha * q) / (alpha - 1), 0.0, 1.0))
    return 0.0


def g_map(x: float, alpha: float, p: float, q: float) -> float:
    """Vectorised one-step map."""
    return (x * alpha * (1 - q) + (1 - x) * p) / (1 + x * (alpha - 1))


def delta(x: float, alpha: float, p: float, q: float) -> float:
    """Net change Δ(x) = g(x) - x."""
    return g_map(x, alpha, p, q) - x


def find_roots_on_unit_interval(
    f: Callable[[float], float], xs: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Bracket and refine roots of f on [0, 1] via brentq."""
    roots: list[float] = []
    for a, b in zip(xs[:-1], xs[1:]):
        fa, fb = f(a), f(b)
        if fa == 0:
            roots.append(float(a))
        elif fa * fb < 0:
            try:
                roots.append(float(brentq(f, a, b)))
            except ValueError:
                pass
    if not roots:
        return np.array([])
    roots_arr = np.array(sorted(roots))
    uniq: list[float] = [roots_arr[0]]
    for r in roots_arr[1:]:
        if abs(r - uniq[-1]) > 1e-4:
            uniq.append(r)
    return np.array(uniq)


def is_stable_equilibrium(f: Callable[[float], float], r: float) -> bool:
    """Return True if r is a stable equilibrium of f (derivative < 0)."""
    eps = 1e-6
    slope = (f(r + eps) - f(r - eps)) / (2 * eps)
    return slope < 0
