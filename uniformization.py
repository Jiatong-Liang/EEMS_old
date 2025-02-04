import jax.numpy as jnp
import jax
import jax.experimental.sparse as jesp
import scipy.sparse as sp
from jax.experimental import sparse
from jax.experimental.sparse import BCOO
from jax import lax
import scipy
from jax import vmap

def _matvec(y, Q, Q1, q):
    n = q.shape[0]
    yn = y[:-1].reshape(n, n)
    yd = jnp.diag(yn)
    ret = Q @ yn + yn @ Q
    ret -= Q1 * yn
    ret -= jnp.diag(q * yd)
    return jnp.append(ret.reshape(n ** 2), q.dot(yd))

def matvec(y, Q, Q1, q):
    return _matvec(jnp.array(y), Q, Q1, q)

@jax.remat
def P(v, Q, Q1, q, Lambda):
    return v + matvec(v, Q, Q1, q)/Lambda

def expm_mult_unif(v0, Q, Q1, q, t):
    # Lambda = 2*(Q.sum(1).todense().max() + q.max()) 
    # two people can migrate with 4 edges with max migration 0.1, hence first term and 2 * 1 for max coalescent
    # rate of two people on a graph.
    Lambda = 2*(0.1 * 4) + 2
    # don't need jax.scipy because none of the arguments are related to jax
    N = scipy.stats.poisson.isf(1e-5, 30 * Lambda)
    # N = 200
    w = v0
    ret = 0 * v0

    @jax.remat
    def f(accum, i):
        w, ret = accum
        ret += jax.scipy.stats.poisson.pmf(i, t * Lambda) * w
        w = P(w, Q, Q1, q, Lambda)
        # This needs to return two arguments so we make the second one to be None
        return (w, ret), None

    (w, ret), _ = jax.lax.scan(f, (w, ret), jnp.arange(1+N))

    return ret[-1]

def unit_vector(size, index):
    # Vector of zeros
    tmp = jnp.zeros(size)
    # ith element will be 1
    # JAX arrays are immutable so we cannot do tmp[index] = 1
    tmp = tmp.at[index].set(1)
    return tmp

# Instead using a coalesce rate c like migration_process jupyter notebook we use q
def solve_ode(time_discretization, init_vertices, q, m, BCOO_indices):
    # edges is 0 ONLY IF you have a graph with only a SINGLE node
    num_nodes = len(q)
    shape = (num_nodes, num_nodes)
    if len(m) == 0:
        # If the graph has absolutely no edges then the graph has 0 for every edge weight
        # in the way we code Q.
        copy = jesp.BCOO(((jnp.array([], dtype=jnp.float32), jnp.empty((0, len(shape)), dtype=jnp.int32))), shape=shape)
    else:
        copy = jesp.BCOO((m.astype(float), BCOO_indices.astype(jnp.int64)), shape=shape)
        
    Q = copy + copy.transpose()
    Q1 = Q.sum(0).todense()[:, None] + Q.sum(1).todense()[None, :]    
    e_i = unit_vector(num_nodes, init_vertices[0])
    e_j = unit_vector(num_nodes, init_vertices[1])
    tmp = (e_i.reshape(-1, 1)) @ jnp.array([e_j])
    y0 = jnp.append(jnp.ravel(tmp), 0)
    sol = vmap(expm_mult_unif, in_axes = (None, None, None, None, 0))(y0, Q, Q1, q, time_discretization)
    probabilities = jnp.append(jnp.diff(sol), jnp.array([1 - sol[-1]]))
    return probabilities, sol

