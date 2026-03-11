import numpy as np
import hphp.analytical


# ── get_x_T_prime ─────────────────────────────────────────────────────────────

def test_get_x_T_prime_fixed_point():
    """x* should be a fixed point of the map."""
    alpha, p, q = 1.1, 0.05, 0.07
    x_star = hphp.analytical.x_star_equilibrium(alpha, p, q)
    x_prime = hphp.analytical.get_x_T_prime(x_star, alpha, p, q)
    assert np.isclose(x_star, x_prime, atol=1e-8)


def test_get_x_T_prime_p_zero_q_zero_alpha_one():
    """When α=1, p=0, q=0 the map is identity."""
    for x in (0.0, 0.3, 0.7, 1.0):
        assert np.isclose(hphp.analytical.get_x_T_prime(x, alpha=1.0, p=0.0, q=0.0), x)


def test_get_x_T_prime_p_zero_no_influx():
    """With p=0 and x=0 the state stays at 0."""
    assert np.isclose(hphp.analytical.get_x_T_prime(0.0, alpha=1.5, p=0.0, q=0.2), 0.0)


def test_get_x_T_prime_q_zero_no_healing():
    """With q=0 and x=1 the state stays at 1."""
    assert np.isclose(hphp.analytical.get_x_T_prime(1.0, alpha=1.5, p=0.0, q=0.0), 1.0)


def test_get_x_T_prime_clip_keeps_in_unit_interval():
    """clip=True must return a value in [0, 1]."""
    result = hphp.analytical.get_x_T_prime(0.99, alpha=2.0, p=0.5, q=0.0, clip=True)
    assert 0.0 <= result <= 1.0


def test_get_x_T_prime_alpha_one_boundary():
    """α=1 simplifies the map; result should still be in [0,1]."""
    result = hphp.analytical.get_x_T_prime(0.5, alpha=1.0, p=0.1, q=0.1)
    assert 0.0 <= result <= 1.0


# ── simulation ────────────────────────────────────────────────────────────────

def test_simulation_generator_length():
    """Generator yields number_of_iterations + 1 values."""
    n = 20
    result = list(hphp.analytical.simulation(0.5, alpha=1.1, p=0.05, q=0.07,
                                              number_of_iterations=n))
    assert len(result) == n + 1


def test_simulation_first_value_is_x0():
    x0 = 0.42
    result = list(hphp.analytical.simulation(x0, alpha=1.1, p=0.0, q=0.0,
                                              number_of_iterations=5))
    assert np.isclose(result[0], x0)


def test_simulation_convergence():
    """Long run from two different initial conditions should converge to x*."""
    alpha, p, q = 1.1, 0.05, 0.07
    x_star = hphp.analytical.x_star_equilibrium(alpha, p, q)
    high = list(hphp.analytical.simulation(0.95, alpha, p, q, number_of_iterations=500))
    low  = list(hphp.analytical.simulation(0.05, alpha, p, q, number_of_iterations=500))
    assert np.isclose(high[-1], x_star, atol=1e-4)
    assert np.isclose(low[-1],  x_star, atol=1e-4)


# ── x_star_equilibrium ────────────────────────────────────────────────────────

def test_x_star_no_influx_no_healing_below_threshold():
    """p=0, α*(1-q) <= 1 => x*=0."""
    assert np.isclose(hphp.analytical.x_star_equilibrium(alpha=1.0, p=0.0, q=0.0), 0.0)


def test_x_star_no_influx_above_threshold():
    """p=0, α*(1-q) > 1 => x* = 1 - αq/(α-1)."""
    alpha, q = 1.5, 0.1
    expected = 1.0 - (alpha * q) / (alpha - 1)
    result = hphp.analytical.x_star_equilibrium(alpha=alpha, p=0.0, q=q)
    assert np.isclose(result, expected)


def test_x_star_positive_p_in_unit_interval():
    """With p>0 the equilibrium should lie strictly inside (0,1)."""
    x_star = hphp.analytical.x_star_equilibrium(alpha=1.1, p=0.05, q=0.07)
    assert 0.0 < x_star < 1.0


def test_x_star_is_fixed_point():
    """x* must satisfy g(x*) = x*."""
    for alpha, p, q in [(1.1, 0.05, 0.07), (1.5, 0.0, 0.1), (1.2, 0.1, 0.0)]:
        x_star = hphp.analytical.x_star_equilibrium(alpha, p, q)
        g_val = hphp.analytical.g_map(x_star, alpha, p, q)
        assert np.isclose(g_val, x_star, atol=1e-7), (alpha, p, q, x_star)


# ── find_roots_on_unit_interval ───────────────────────────────────────────────

def test_find_roots_recovers_known_root():
    """f(x) = x - 0.3 has a root at 0.3."""
    xs = np.linspace(0, 1, 1000)
    roots = hphp.analytical.find_roots_on_unit_interval(lambda x: x - 0.3, xs)
    assert len(roots) == 1
    assert np.isclose(roots[0], 0.3, atol=1e-5)


def test_find_roots_quadratic():
    """(x-0.2)(x-0.7) has two roots on [0,1]."""
    xs = np.linspace(0, 1, 2000)
    roots = hphp.analytical.find_roots_on_unit_interval(
        lambda x: (x - 0.2) * (x - 0.7), xs
    )
    assert len(roots) == 2
    assert np.isclose(roots[0], 0.2, atol=1e-4)
    assert np.isclose(roots[1], 0.7, atol=1e-4)


def test_find_roots_no_root_returns_empty():
    """f(x) = 1 has no roots."""
    xs = np.linspace(0, 1, 500)
    roots = hphp.analytical.find_roots_on_unit_interval(lambda x: 1.0, xs)
    assert len(roots) == 0


def test_find_roots_matches_x_star():
    """Roots of delta should match x_star_equilibrium."""
    alpha, p, q = 1.1, 0.05, 0.07
    xs = np.linspace(0, 1, 2000)
    f = lambda x: hphp.analytical.delta(x, alpha, p, q)
    roots = hphp.analytical.find_roots_on_unit_interval(f, xs)
    x_star = hphp.analytical.x_star_equilibrium(alpha, p, q)
    assert any(np.isclose(r, x_star, atol=1e-4) for r in roots)


# ── is_stable_equilibrium ─────────────────────────────────────────────────────

def test_is_stable_equilibrium_stable():
    """x* for (α=1.1, p=0.05, q=0.07) should be stable."""
    alpha, p, q = 1.1, 0.05, 0.07
    x_star = hphp.analytical.x_star_equilibrium(alpha, p, q)
    f = lambda x: hphp.analytical.delta(x, alpha, p, q)
    assert hphp.analytical.is_stable_equilibrium(f, x_star) is True


def test_is_stable_equilibrium_unstable():
    """x=0 for (α=1.5, p=0, q=0.1) is unstable (derivative > 0 there)."""
    alpha, p, q = 1.5, 0.0, 0.1
    f = lambda x: hphp.analytical.delta(x, alpha, p, q)
    # x=0 is a root of delta when p=0; α*(1-q)=1.35>1 so it is unstable
    assert hphp.analytical.is_stable_equilibrium(f, 0.0) is False
