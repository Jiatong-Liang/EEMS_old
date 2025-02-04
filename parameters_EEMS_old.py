"Different parameterizations needed for MCMC and HMM"
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax_dataclasses as jdc

import phlash.size_history
import phlash.transition
from phlash.util import Pattern, softplus_inv
from jax.nn import sigmoid

# from solving_ode_EEMS_old import solve_ode
# from discrete_approx import solve_ode
from uniformization import solve_ode

def from_pmf(t, p):
    """Initialize a size history from a distribution function of coalescent times.

    Args:
        t: time points
        p: p[i] = probability of coalescing in [t[i], t[i + 1])
    """
    # sum_i = 0.0
    # c = []
    # p = jnp.clip(p, 0, jnp.inf)
    # for dt, p_i in zip(jnp.diff(t), p[:-1]):
    #     x = p_i / (1 - sum_i)
    #     # bad = jnp.logical_or(jnp.logical_or(x >= 1, x <= 0), jnp.logical_or(jnp.isnan(x), jnp.isinf(x)))
    #     bad = (x >= 1) | (x <= 0) | jnp.isnan(x) | jnp.isinf(x)
    #     x_safe = jnp.where(bad, 0., x)
    #     c_safe = jnp.where(bad, 1e-4, -jnp.log1p(-x_safe) / dt)
    #     c.append(c_safe)
    #     sum_i = sum_i + p_i
    # # coalescent rate in last period is not identifiable from this data.
    # c.append(c[-1])
    sum_initial = 0.0
    c = []
    p = jnp.clip(p, 0, jnp.inf)
    tol = 1e-4
    # difference in times
    dts = jnp.diff(t)
    p_truncated = p[:-1]

    # scan function
    def scan_fn(carry, inputs):
        sum_i = carry
        dt, p_i = inputs
        x = p_i / (1 - sum_i)
        bad = (x >= 1) | (x <= 0) | jnp.isnan(x) | jnp.isinf(x)
        x_safe = jnp.where(bad, 0., x)
        c_safe = jnp.where(bad, tol, -jnp.log1p(-x_safe) / dt)
        c_final = jnp.where(c_safe > tol, c_safe, tol)
        sum_i = sum_i + p_i
        return sum_i, c_final
    
    inputs = (dts, p_truncated)
    sum_final, c = jax.lax.scan(scan_fn, sum_initial, inputs)
    # Append the last coalescent rate (not identifiable from data)
    c = jnp.append(c, c[-1])
    return jnp.array(t), jnp.array(c)

def logit(z):
    """Compute the logit (inverse sigmoid) of z."""
    z = jnp.clip(z, a_min=1e-7, a_max=1 - 1e-7)  # Avoid numerical instability
    return jnp.log(z / (1 - z))

# t_tr is time discretization that will be randomized
# m_tr and q_tr are what we attempt to estimate, they are transformed with softplus
@jdc.pytree_dataclass
class MCMCParams:
    pattern: jdc.Static[str]
    t_tr: jax.Array
    # t_tr: jdc.Static[jax.Array]
    BCOO_indices: jdc.Static[jax.Array]
    m_tr: jax.Array
    q_tr: jax.Array
    # m_tr: jdc.Static[jax.Array]
    # q_tr: jdc.Static[jax.Array]
    rho_over_theta_tr: float
    theta: jdc.Static[float]
    alpha: jdc.Static[float]
    beta: jdc.Static[float]

    @classmethod
    def from_linear(
        cls,
        pattern: str,
        t1: float,
        tM: float,
        BCOO_indices: jax.Array,
        m: jax.Array,
        q: jax.Array,
        theta: float,
        rho: float,
        alpha: float = 0.0,
        beta: float = 0.0,
    ) -> "MCMCParams":
        dtM = tM - t1
        t_tr = jnp.array([jnp.log(t1), jnp.log(dtM)])
        # t_tr = jnp.array([softplus_inv(t1), softplus_inv(dtM)])
        rho_over_theta_tr = jsp.special.logit((rho / theta - 0.1) / 9.9)
        return cls(
            pattern=pattern,
            BCOO_indices = BCOO_indices,
            # m_tr = jnp.array(logit(m)),
            # q_tr = jnp.array(logit(q)),
            m_tr=jnp.array(softplus_inv(m)),
            q_tr=jnp.array(softplus_inv(q)),
            t_tr=t_tr,
            rho_over_theta_tr=rho_over_theta_tr,
            theta=theta,
            alpha=alpha,
            beta=beta,
        )

    def to_dm(self, init_vertices) -> phlash.size_history.DemographicModel:
        pat = Pattern(self.pattern)
        t1, tM = self.t
        t = jnp.insert(jnp.geomspace(t1, tM, pat.M - 1), 0, 0.0)
        probabilities, sol = solve_ode(t, init_vertices, self.c, self.m, self.BCOO_indices)
        t, c = from_pmf(t, probabilities)
        eta = phlash.size_history.SizeHistory(t=t, c=c)
        assert eta.t.shape == eta.c.shape
        return phlash.size_history.DemographicModel(
            eta=eta, theta=self.theta, rho=self.rho
        )
    
    @property
    def M(self):
        return Pattern(self.pattern).M

    @property
    def rho_over_theta(self):
        # this transformation ensures that rho/theta is in [.1, 10]
        return 0.1 + 9.9 * jsp.special.expit(self.rho_over_theta_tr)

    @property
    def rho(self):
        return self.rho_over_theta * self.theta

    @property
    def t(self):
        t1, dtM = jnp.exp(self.t_tr)
        # t1, dtM = jax.nn.softplus(self.t_tr)
        tM = t1 + dtM
        return t1, tM

    @property
    def c(self):
        # return 2 * sigmoid(self.q_tr)
        return jax.nn.softplus(self.q_tr)
    
    @property
    def m(self):
        # return 0.1 * sigmoid(self.m_tr)
        return jax.nn.softplus(self.m_tr)
        
    @property
    def log_c(self):
        return jnp.log(self.c)