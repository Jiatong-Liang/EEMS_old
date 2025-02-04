# This file contains methods required solely for running MCMC

from jaxtyping import Array, Float, Float64, Int8, Int64
import phlash
from parameters_EEMS_old import MCMCParams
from phlash.model import log_prior
from phlash.params import PSMCParams
from jax import vmap
import jax
import jax.numpy as jnp
from utility_methods_old import random_chromosome_pair

# def log_density(
#     mcp: MCMCParams,
#     # init_vertices: tuple[int, int],
#     init_vertices: Array,
#     inds: Int64[Array, "batch"],
#     warmup: Int8[Array, "c ell"],
#     c: Float64[Array, "3"],
#     kern: "phlash.gpu.PSMCKernel",
# ) -> float:
#     r"""
#     Computes the log density of a statistical model by combining the contributions from
#     the prior, the hidden Markov model (HMM), and the allele frequency spectrum (AFS)
#     model, weighted by given coefficients.

#     Args:
#         mcp: The Markov Chain Monte Carlo parameters used to specify the model.
#         c: Weights for each component of the density - prior, HMM model, and AFS model.
#         inds: Mini-batch indices for selecting subsets of the data.
#         data: Data matrix used in the model computation.
#         kern: An instantiated PSMC Kernel used in the computation.
#         afs: The allele frequency spectrum data.
#         use_folded_afs: Whether to fold the afs, if ancestral allele is not known.

#     Returns:
#         The log density, or negative infinity where the result is not finite.
#     """
#     # warmup = warmup[inds]
#     dm = mcp.to_dm(init_vertices)
#     pp = PSMCParams.from_dm(dm)
#     pis = vmap(lambda pp, d: phlash.hmm.psmc_ll(pp, d)[0], (None, 0))(
#         pp, warmup
#     )
#     pps = vmap(lambda pi: pp._replace(pi=pi))(pis)
#     # l1 = log_prior(mcp)
#     l2 = vmap(kern.loglik, (0, 0))(pps, inds).sum()
#     return l2

def log_density(
    mcp: MCMCParams,
    # init_vertices: tuple[int, int],
    init_vertices: Array,
    inds: Int64[Array, "batch"],
    warmup: Int8[Array, "c ell"],
    c: Float64[Array, "3"],
    kern: "phlash.gpu.PSMCKernel",
) -> float:
    r"""
    Computes the log density of a statistical model by combining the contributions from
    the prior, the hidden Markov model (HMM), and the allele frequency spectrum (AFS)
    model, weighted by given coefficients.

    Args:
        mcp: The Markov Chain Monte Carlo parameters used to specify the model.
        c: Weights for each component of the density - prior, HMM model, and AFS model.
        inds: Mini-batch indices for selecting subsets of the data.
        data: Data matrix used in the model computation.
        kern: An instantiated PSMC Kernel used in the computation.
        afs: The allele frequency spectrum data.
        use_folded_afs: Whether to fold the afs, if ancestral allele is not known.

    Returns:
        The log density, or negative infinity where the result is not finite.
    """
    l2_sum = 0
    # carry are the inputs you carry over from iteration to iteration
    # inputs are what you map over and replaces things like init_vertices[i] or inds[i]
    def scan_body(carry, inputs):
        l2_sum, warmup = carry
        inds_tmp, init_vertices_tmp = inputs
        # Access current indices and vertices
        # inds_tmp = inds[i]
        # init_vertices_tmp = init_vertices[i]

        # Apply operations involving inds and init_vertices
        warmup_tmp = warmup[inds_tmp]
        dm = mcp.to_dm(init_vertices_tmp)
        pp = PSMCParams.from_dm(dm)
        pis = vmap(lambda pp, d: phlash.hmm.psmc_ll(pp, d)[0], (None, 0))(
            pp, warmup_tmp
        )
        pps = vmap(lambda pi: pp._replace(pi=pi))(pis)
        # l1 = log_prior(mcp)
        l2 = vmap(kern.loglik, (0, 0))(pps, inds_tmp).sum()
        l2_sum += l2

        # Return updated carry
        return (l2_sum, warmup), None

    # Initialize the carry with starting values of l2_sum and warmup
    carry = (0, warmup)

    # Iterate over the zipped (inds, init_vertices), there's no specific value like 3 or 5
    # you iterate over, you simply iterate until all pairs of (inds, init_vertices) are exhausted
    carry, _ = jax.lax.scan(scan_body, carry, (inds, init_vertices))
    # carry, _ = jax.lax.scan(scan_body, carry, jnp.arange(3))

    # Return the final l2_sum and ignore warmup
    l2_sum, _ = carry
    return l2_sum

### This needs to be cleaned up, some intializations are just literally fixed
def running_MCMC(initialization, TreeSeq, pairs, mapping, num_particles):
    # Changed mcp to init and test_kern to train_kern
    import numpy as np
    import jax.numpy as jnp
    import equinox as eqx
    from jax import grad

    import optax
    import blackjax
    # from my_svgd2 import my_svgd

    rate = 0.4
    opt = optax.amsgrad(learning_rate=rate)
    # svgd = my_svgd(eqx.filter_grad(log_density), opt)
    svgd = blackjax.svgd(eqx.filter_grad(log_density), opt)

    from jax.flatten_util import ravel_pytree
    import jax

    key = jax.random.PRNGKey(1)
    M = initialization.M
    x0, unravel = ravel_pytree(initialization)
    ndim = len(x0)
    sigma = 1.0
    prior_mu = x0
    prior_prec = sigma * jnp.eye(ndim)  # Exclude the first two entries
    num_samples = num_particles
    key, rng_key_init = jax.random.split(key, 2)

    from jax import vmap
    initial_particles = vmap(unravel)(
            jax.random.multivariate_normal(
                rng_key_init,
                prior_mu,
                prior_prec,
                shape=(num_samples,),
            )
        )
        
    ####################

    from loguru import logger
    from phlash.data import Contig, init_mcmc_data
    from phlash.data import TreeSequenceContig

    # Inserting a bunch of initial values
    niter = 2000
    window_size = 100
    overlap = 500 # 500
    chunk_size = 20000 # 200
    max_samples = 20
    num_workers = 1

    logger.info("Loading data")

    # 1 by 2 nodes with 8 SS
    # pairs = [(0,1), (16,17), (2,18), (19,3)]
    # 1 by 5 nodes, with 8 SS
    # pairs = [(0,16), (1,32), (48,2), (4,64), (17,33), (18, 49), (19,65), (34,50), (35, 66), (51,67), (13,15), (12,14), (30,31), (28,29), (45,44), (47,46), (60, 61), (63, 62), (76,78), (77,79)]
    # for 2 by 5 number of nodes
    # pairs = [(0, 1), (16, 17), (32, 33), (48, 49), (64, 65), (80, 81), (96, 97), (112, 113), (128, 129), (144, 145), (2, 18), (3, 34), (4, 50), (5, 66), (6, 82), (7, 98), (8, 114), (9, 130), (10, 146), (19, 35), (20, 51), (21, 67), (22, 83), (23, 99), (24, 115), (25, 131), (26, 147), (36, 52), (37, 68), (38, 84), (39, 100), (40, 116), (41, 132), (42, 148), (53, 69), (54, 85), (55, 101), (56, 117), (57, 133), (58, 149), (70, 86), (71, 102), (72, 118), (73, 134), (74, 150), (87, 103), (88, 119), (89, 135), (90, 151), (104, 120), (105, 136), (106, 152), (121, 137), (122, 153), (138, 154)]
    # pairs = [(0,15), (1,16), (2,32), (3,48), (4,64),(18,19),(20,33),(21,49),(22,65),(34,35),(36,50),(37,66),(51,52),(53,69),(5,7),(6,8),(10,13),(25, 27),(28,26),(31,29),(42,40),(43,44),(47,46),(59,56),(57,58),(62,60),(63,61),(71,77),(72,75),(79,74),(76,78)]
    # for 2 by 2 with 8 SS
    # pairs = [(0,1), (16,17), (32,33), (48,49), (2, 18), (19, 34), (35, 50), (51, 3)    , (4,47), (5,63), (30,59), ]
    # pairs = [(0,1), (16,17), (32,33), (48,49), (2, 18), (19, 34), (35, 50), (51, 3)]
    data = [TreeSeq] 

    afs, chunks = init_mcmc_data(
        data, window_size, overlap, chunk_size, max_samples, num_workers
    )

    S = 5
    N = len(chunks)

    warmup_chunks, data_chunks = np.split(chunks, [overlap], axis=1)

    # the warmup chunks and data chunks are analyzed differently; the data chunks load
    # onto the GPU whereas the warmup chunks are processed by native jax.
    from phlash.kernel import get_kernel
    train_kern = get_kernel(
        M=M,
        data=np.ascontiguousarray(data_chunks),
        # double_precision=options.get("double_precision", False),
        double_precision=False,
    )

    # if there is a test set, define elpd() function for computing expected
    # log-predictive density. used to gauge convergence.
    test_data = None
    if test_data:
        d = test_data.get_data(window_size)
        test_afs = d["afs"]
        test_data = d["het_matrix"][:max_samples]
        N_test = test_data.shape[0]
        test_kern = get_kernel(
            M=M,
            data=np.ascontiguousarray(d["het_matrix"]),
            double_precision=False,
        )

        @jit
        def elpd(mcps):
            @vmap
            def _elpd_ll(mcp):
                return log_density(
                    mcp,
                    c=jnp.array([0.0, 1.0, 1.0]),
                    inds=jnp.arange(N_test),
                    kern=test_kern,
                    warmup=jnp.full([N_test, 1], -1, dtype=jnp.int8),
                    # afs=test_afs,
                    # afs_transform=afs_transform,
                )

            return _elpd_ll(mcps).mean()

    # to have unbiased gradient estimates, need to pre-multiply the chunk term by ratio
    # (dataset size) / (minibatch size) = N / S.
    kw = dict(
        kern=train_kern,
        c=jnp.array([1.0, N / S, 1.0]),
        # afs=afs,
        # afs_transform=afs_transform,
    )

    # build the plot callback
    # cb = options.get("callback")
    cb = None
    truth = None
    if not cb:
        try:
            from phlash.liveplot import liveplot_cb

            # cb = liveplot_cb(truth=options.get("truth"))
            cb = liveplot_cb(truth=truth)
        except ImportError:
            # if necessary libraries aren't installed, just initialize a dummy callback
            def cb(*a, **kw):
                pass

    from phlash.size_history import DemographicModel
    def dms():
        ret = vmap(MCMCParams.to_dm)(state.particles)
        # rates are per window, so we have to scale up to get the per-base-pair rates.
        ret = ret._replace(theta=ret.theta / window_size, rho=ret.rho / window_size)
        if mutation_rate:
            ret = vmap(DemographicModel.rescale, (0, None))(ret, mutation_rate)
        return ret

    ema = best_elpd = None
    a = 0  # tracks the number of iterations since the elpd went up
    global _particles  # for debugging

    import tqdm
    
    #########
    from jax import jit
    state = svgd.init(eqx.filter(initial_particles, eqx.is_inexact_array))
    # state = svgd.init(eqx.filter(initial_particles, eqx.is_inexact_array), {"length_scale": 1.0}, num_particles, vertex_to_chunk_map)
    # this function takes gradients steps.
    step = jit(svgd.step, static_argnames=["kern"])

    tmp2 = []
    tmp3 = []

    key = jax.random.PRNGKey(3)
    with tqdm.trange(
    # niter, disable=not options.get("progress", True), desc="Fitting model"
    niter, disable=not True, desc="Fitting model"
    ) as pbar:
        for i in pbar:
            key, subkey = jax.random.split(key, 2)
            # random_keys = random_chromosome_pair(pairs, 1)
            # inds_unsorted = jnp.array((chunk_map[random_keys[0]]))
            # inds = kw["inds"] = inds_unsorted
            # kw["init_vertices"] = vertex_map[random_keys[0]]
            # kw["warmup"] = warmup_chunks[inds]

            number = 3
            random_keys = random_chromosome_pair(pairs, number)
            kw["init_vertices"] = jnp.array(random_keys)
            kw["inds"] = jnp.array([mapping[random_keys[i]] for i in range(number)])
            kw["warmup"] = warmup_chunks

            # print(state.particles)
            with jax.debug_nans(False), jax.disable_jit(False):
                state1 = step(state, **kw)

            def check_finite(x):
                assert jnp.isfinite(x).all()
                return x

            state = jax.tree.map(check_finite, state1)
            _particles = state.particles
            print(np.median(jax.nn.softplus(_particles.m_tr), axis = 0))
            print(np.median(jax.nn.softplus(_particles.q_tr), axis = 0))
            # print(np.median(0.1 * jax.nn.sigmoid(_particles.m_tr), axis = 0))
            # print(np.median(2 * jax.nn.sigmoid(_particles.m_tr), axis = 0))

            if test_data is not None and i % 10 == 0:
                e = elpd(state.particles)
                if ema is None:
                    ema = e
                else:
                    ema = 0.9 * ema + 0.1 * e
                if best_elpd is None or ema > best_elpd[1]:
                    a = 0
                    best_elpd = (i, ema, state)
                else:
                    a += 1
                if i - best_elpd[0] > elpd_cutoff:
                    logger.info(
                        "The expected log-predictive density has not improved in "
                        f"the last {elpd_cutoff} iterations; exiting."
                    )
                    break
                pbar.set_description(f"elpd={ema:.2f} a={a}")
            # cb(dms())
    logger.info("MCMC finished successfully")
    # notify the live plot that we are done. fails if we are not using liveplot.
    return _particles
