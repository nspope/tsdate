# MIT License
#
# Copyright (c) 2021-23 Tskit Developers
# Copyright (c) 2020-21 University of Oxford
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
Tools for approximating combinations of Gamma variates with Gamma distributions
"""
import logging

import mpmath
import numba
import numpy as np
from mpmath.libmp.libhyper import NoConvergence as MPNoConvergence

from . import hypergeo

# TODO: these are reasonable defaults but could
# be set via a control dict
_KLMIN_MAXITER = 100
_KLMIN_TOL = np.sqrt(np.finfo(np.float64).eps)


class KLMinimizationFailed(Exception):
    pass


@numba.njit("UniTuple(float64, 3)(float64, float64)")
def approximate_log_moments(mean, variance):
    """
    Approximate log moments via a second-order Taylor series expansion around
    the mean, e.g.:

      E[f(x)] \\approx E[f(mean)] + variance * f''(mean)/2

    Returns approximations to E[log x], E[x log x], E[(log x)^2]
    """
    assert mean > 0
    assert variance > 0
    logx = np.log(mean) - 0.5 * variance / mean**2
    xlogx = mean * np.log(mean) + 0.5 * variance / mean
    logx2 = np.log(mean) ** 2 + (1 - np.log(mean)) * variance / mean**2
    return logx, xlogx, logx2


@numba.njit("UniTuple(float64, 2)(float64, float64)")
def approximate_gamma_kl(x, logx):
    """
    Use Newton root finding to get gamma parameters matching the sufficient
    statistics :math:`E[x]` and :math:`E[\\log x]`, minimizing KL divergence.

    The initial condition uses the upper bound :math:`digamma(x) \\leq log(x) - 1/2x`.

    Returns the shape and rate of the approximating gamma.
    """
    assert np.isfinite(x) and np.isfinite(logx)
    if not np.log(x) > logx:
        raise KLMinimizationFailed("log E[t] <= E[log t] violates Jensen's inequality")
    alpha = 0.5 / (np.log(x) - logx)  # lower bound on alpha
    # asymptotically the lower bound becomes sharp
    if 1.0 / alpha < 1e-4:
        return alpha, alpha / x
    itt = 0
    delta = np.inf
    # determine convergence when the change in alpha falls below
    # some small value (e.g. square root of machine precision)
    while np.abs(delta) > alpha * _KLMIN_TOL:
        if itt > _KLMIN_MAXITER:
            raise KLMinimizationFailed("Maximum iterations reached in KL minimization")
        delta = hypergeo._digamma(alpha) - np.log(alpha) + np.log(x) - logx
        delta /= hypergeo._trigamma(alpha) - 1 / alpha
        alpha -= delta
        itt += 1
    if not np.isfinite(alpha) or alpha <= 0:
        raise KLMinimizationFailed("Invalid shape parameter in KL minimization")
    return alpha, alpha / x


@numba.njit("UniTuple(float64, 2)(float64, float64)")
def approximate_gamma_mom(mean, variance):
    """
    Use the method of moments to approximate a distribution with a gamma of the
    same mean and variance
    """
    assert mean > 0
    assert variance > 0
    alpha = mean**2 / variance
    beta = mean / variance
    return alpha, beta


def rescale_gamma(posterior, edges_in, edges_out, new_shape):
    """
    Given a factorization of gamma parameters in `posterior` into additive
    terms `edges_in` and `edges_out` and a prior, rescale so that the posterior
    shape has a fixed value.
    """

    in_shape, in_rate = edges_in[:, 0], edges_in[:, 1]
    out_shape, out_rate = edges_out[:, 0], edges_out[:, 1]
    post_shape, post_rate = posterior[0], posterior[1]

    assert post_shape > 0 and post_rate > 0

    # new posterior parameters
    new_rate = new_shape * post_rate / post_shape

    # rescale messages to match desired shape
    shape_scale = (new_shape - 1) / (post_shape - 1)
    rate_scale = new_rate / post_rate
    in_shape = (in_shape - 1) * shape_scale + 1
    out_shape = (out_shape - 1) * shape_scale + 1
    in_rate = in_rate * rate_scale
    out_rate = out_rate * rate_scale

    return (
        np.array([new_shape, new_rate]),
        np.column_stack([in_shape, in_rate]),
        np.column_stack([out_shape, out_rate]),
    )


@numba.njit("UniTuple(float64, 2)(float64[:], float64[:])")
def average_gammas(shape, rate):
    """
    Given shape and rate parameters for a set of gammas, average sufficient
    statistics so as to get a "global" gamma
    """
    assert shape.size == rate.size, "Array sizes are not equal"
    avg_x = 0.0
    avg_logx = 0.0
    for a, b in zip(shape, rate):
        avg_logx += hypergeo._digamma(a) - np.log(b)
        avg_x += a / b
    avg_x /= shape.size
    avg_logx /= shape.size
    return approximate_gamma_kl(avg_x, avg_logx)


@numba.njit(
    "UniTuple(float64, 5)(float64, float64, float64, float64, float64, float64)"
)
def sufficient_statistics(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    """
    Calculate gamma sufficient statistics for the PDF proportional to
    :math:`Ga(t_j | a_j, b_j) Ga(t_i | a_i, b_i) Po(y_{ij} |
    \\mu_{ij} t_i - t_j)`, where :math:`i` is the parent and :math:`j` is
    the child.

    :param float a_i: the shape parameter of the cavity distribution for the parent
    :param float b_i: the rate parameter of the cavity distribution for the parent
    :param float a_j: the shape parameter of the cavity distribution for the child
    :param float b_j: the rate parameter of the cavity distribution for the child
    :param float y_ij: the number of mutations on the edge
    :param float mu_ij: the span-weighted mutation rate of the edge

    :return: normalizing constant, E[t_i], E[log t_i], E[t_j], E[log t_j]
    """

    a = a_i + a_j + y_ij
    b = a_j
    c = a_j + y_ij + 1
    t = mu_ij + b_i
    z = (mu_ij - b_j) / t

    if not (a > 0 and b > 0 and c > 0 and t > 0):  # skip update
        raise Exception("Negative parameters")

    log_f, sign_f, da_i, db_i, da_j, db_j = hypergeo._hyp2f1(
        a_i, b_i, a_j, b_j, y_ij, mu_ij
    )

    logconst = (
        log_f + hypergeo._betaln(y_ij + 1, b) + hypergeo._gammaln(a) - a * np.log(t)
    )

    t_i = -db_i + a / t
    t_j = -db_j
    ln_t_i = da_i - np.log(t) + hypergeo._digamma(a)
    ln_t_j = (
        da_j
        - np.log(t)
        + hypergeo._digamma(a)
        + hypergeo._digamma(b)
        - hypergeo._digamma(c)
    )

    return logconst, t_i, ln_t_i, t_j, ln_t_j


def mean_and_variance(a_i, b_i, a_j, b_j, y_ij, mu_ij, dps=100, maxterms=1e4):
    """
    Calculate mean and variance for the PDF proportional to
    :math:`Ga(t_j | a_j, b_j) Ga(t_i | a_i, b_i) Po(y_{ij} |
    \\mu_{ij} (t_i - t_j))`, where :math:`i` is the parent and :math:`j` is
    the child.

    This is intended to provide a stable approximation when calculation of
    gamma sufficient statistics fails (e.g. when the log-normalizer is close to
    singular). Calculations are done at arbitrary precision and are slow.

    :param float a_i: the shape parameter of the cavity distribution for the parent
    :param float b_i: the rate parameter of the cavity distribution for the parent
    :param float a_j: the shape parameter of the cavity distribution for the child
    :param float b_j: the rate parameter of the cavity distribution for the child
    :param float y_ij: the number of mutations on the edge
    :param float mu_ij: the span-weighted mutation rate of the edge
    :param int dps: decimal places for multiprecision computations

    :return: normalizing constant, E[t_i], V[t_i], E[t_j], V[t_j]
    """

    a = a_i + a_j + y_ij
    b = a_j
    c = a_j + y_ij + 1
    t = mu_ij + b_i
    z = (mu_ij - b_j) / t

    if not (a > 0 and b > 0 and c > 0 and t > 0):  # skip update
        raise Exception("Negative parameters")

    # 2F1 and first/second derivatives of argument, in arbitrary precision
    with mpmath.workdps(dps):
        s0 = a * b / c
        s1 = s0 * (a + 1) * (b + 1) / (c + 1)
        v0 = mpmath.hyp2f1(a, b, c, z, maxterms=maxterms)
        v1 = s0 * (mpmath.hyp2f1(a + 1, b + 1, c + 1, z, maxterms=maxterms) / v0)
        v2 = s1 * (mpmath.hyp2f1(a + 2, b + 2, c + 2, z, maxterms=maxterms) / v0)
        logconst = float(mpmath.log(v0))
        dz = float(v1)
        d2z = float(v2)

    # mean / variance of child and parent age
    logconst += hypergeo._betaln(y_ij + 1, b) + hypergeo._gammaln(a) - a * np.log(t)
    t_i = dz * z / t + a / t
    va_t_i = z / t**2 * (d2z * z + 2 * dz * (1 + a)) + a * (1 + a) / t**2 - t_i**2
    t_j = dz / t
    va_t_j = d2z / t**2 - t_j**2

    return logconst, t_i, va_t_i, t_j, va_t_j


def gamma_projection(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    """
    Match a pair of gamma distributions to the potential function
    :math:`Ga(t_j | a_j, b_j) Ga(t_i | a_i, b_i) Po(y_{ij} |
    \\mu_{ij} t_i - t_j)`, where :math:`i` is the parent and :math:`j` is
    the child, by minimizing KL divergence.

    :param float a_i: the shape parameter of the cavity distribution for the parent
    :param float b_i: the rate parameter of the cavity distribution for the parent
    :param float a_j: the shape parameter of the cavity distribution for the child
    :param float b_j: the rate parameter of the cavity distribution for the child
    :param float y_ij: the number of mutations on the edge
    :param float mu_ij: the span-weighted mutation rate of the edge

    :return: gamma parameters for parent and child
    """
    try:
        logconst, t_i, ln_t_i, t_j, ln_t_j = sufficient_statistics(
            a_i, b_i, a_j, b_j, y_ij, mu_ij
        )
        proj_i = approximate_gamma_kl(t_i, ln_t_i)
        proj_j = approximate_gamma_kl(t_j, ln_t_j)
    except (hypergeo.Invalid2F1, KLMinimizationFailed):
        try:
            logging.info(
                f"Matching sufficient statistics failed with parameters: "
                f"{a_i} {b_i} {a_j} {b_j} {y_ij} {mu_ij},"
                f"matching mean and variance instead"
            )
            logconst, t_i, va_t_i, t_j, va_t_j = mean_and_variance(
                a_i, b_i, a_j, b_j, y_ij, mu_ij
            )
            proj_i = approximate_gamma_mom(t_i, va_t_i)
            proj_j = approximate_gamma_mom(t_j, va_t_j)
        except MPNoConvergence:
            raise hypergeo.Invalid2F1(
                "Hypergeometric series does not converge; the approximate "
                "marginal is likely degenerate.  This may reflect a topological "
                "constraint that is at odds with the mutational data. Setting "
                "'max_shape' to a large value (e.g. 1000) will prevent degenerate "
                "marginals, but the results should be treated with care."
            )
    except:  # skip update
        print("skipping", [a_i, b_i, a_j, b_j, y_ij, mu_ij])  # DEBUG
        logconst = np.nan
        proj_i = np.array([a_i, b_i])
        proj_j = np.array([a_j, b_j])

    return logconst, np.array(proj_i), np.array(proj_j)


def mutation_sufficient_statistics(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    r"""
    Calculate gamma sufficient statistics for the PDF proportional to:

    ..math::

        p(x) = \int_0^\infty \int_0^{t_i} Unif(x | t_i, t_j)
        Ga(t_i | a_i, b_i) Ga(t_j | a_j b_j) Po(y | \mu_ij (t_i - t_j)) dt_j dt_i

    which models the time :math:`x` of a mutation uniformly distributed between
    parent age :math:`t_i` and child age :math:`t_j`, on a branch with
    :math:`y_{ij}` mutations and total mutation rate :math:`\mu_{ij}`.

    Returns E[x] and E[\log x].
    """

    # E[t_m]
    f, t_i, _, t_j, _ = sufficient_statistics(a_i, b_i, a_j, b_j, y_ij, mu_ij)
    t_m = t_i / 2 + t_j / 2

    # E[log t_m]
    f_i, _, ln_t_i, _, _ = sufficient_statistics(
        a_i + 1, b_i, a_j, b_j, y_ij - 1, mu_ij
    )
    f_j, _, _, _, ln_t_j = sufficient_statistics(
        a_i, b_i, a_j + 1, b_j, y_ij - 1, mu_ij
    )
    ln_t_m = np.exp(f_j - f) * (1.0 - ln_t_j) - np.exp(f_i - f) * (1.0 - ln_t_i)

    return t_m, ln_t_m


def mutation_mean_and_variance(a_i, b_i, a_j, b_j, y_ij, mu_ij):
    r"""
    Calculate mean and variance of the PDF proportional to:

    ..math::

        p(x) = \int_0^\infty \int_0^{t_i} Unif(x | t_i, t_j)
        Ga(t_i | a_i, b_i) Ga(t_j | a_j b_j) Po(y | \mu_ij (t_i - t_j)) dt_j dt_i

    which models the time :math:`x` of a mutation uniformly distributed between
    parent age :math:`t_i` and child age :math:`t_j`, on a branch with
    :math:`y_{ij}` mutations and total mutation rate :math:`\mu_{ij}`.

    Returns E[x] and V[x].
    """

    # E[t_m]
    # f, t_i, _, t_j, _ = sufficient_statistics(a_i, b_i, a_j, b_j, y_ij, mu_ij)
    f, t_i, _, t_j, _ = mean_and_variance(a_i, b_i, a_j, b_j, y_ij, mu_ij)
    t_m = t_i / 2 + t_j / 2

    # V[t_m]
    # f_i, *_ = sufficient_statistics(a_i + 2, b_i, a_j, b_j, y_ij, mu_ij)
    # f_ij, *_ = sufficient_statistics(a_i + 1, b_i, a_j + 1, b_j, y_ij, mu_ij)
    # f_j, *_ = sufficient_statistics(a_i, b_i, a_j + 2, b_j, y_ij, mu_ij)
    f_i, *_ = mean_and_variance(a_i + 2, b_i, a_j, b_j, y_ij, mu_ij)
    f_ij, *_ = mean_and_variance(a_i + 1, b_i, a_j + 1, b_j, y_ij, mu_ij)
    f_j, *_ = mean_and_variance(a_i, b_i, a_j + 2, b_j, y_ij, mu_ij)
    va_t_m = 1 / 3 * (np.exp(f_i - f) + np.exp(f_ij - f) + np.exp(f_j - f)) - t_m**2

    return t_m, va_t_m


def mutation_gamma_projection(a_i, b_i, a_j, b_j, y_ij, mu_ij, leaf=False):
    r"""
    Match a gamma distribution via KL minimization to the potential function

    ..math::

        p(x) = \int_0^\infty \int_0^{t_i} Unif(x | t_i, t_j)
        Ga(t_i | a_i, b_i) Ga(t_j | a_j b_j) Po(y | \mu_ij (t_i - t_j)) dt_j dt_i

    which models the time :math:`x` of a mutation uniformly distributed between
    parent age :math:`t_i` and child age :math:`t_j`, on a branch with
    :math:`y_{ij}` mutations and total mutation rate :math:`\mu_{ij}`.

    :param float a_i: the shape parameter of the cavity distribution for the parent
    :param float b_i: the rate parameter of the cavity distribution for the parent
    :param float a_j: the shape parameter of the cavity distribution for the child
    :param float b_j: the rate parameter of the cavity distribution for the child
    :param float y_ij: the number of mutations on the edge
    :param float mu_ij: the span-weighted mutation rate of the edge

    :return: gamma parameters for mutation age
    """
    if leaf:
        # E[x] = int_0^inf int_0^t x/t Ga(t | a, b) dx dt = E[t] / 2
        # E[ln x] = int_0^inf int_0^t log(x)/t Ga(t | a, b) dx dt = E[log t] - 1.0
        t = 0.5 * (y_ij + a_i) / (mu_ij + b_i)
        ln_t = hypergeo._digamma(y_ij + a_i) - np.log(mu_ij + b_i) - 1.0
        proj_t = approximate_gamma_kl(t, ln_t)
    else:
        try:
            t, ln_t = mutation_sufficient_statistics(a_i, b_i, a_j, b_j, y_ij, mu_ij)
            proj_t = approximate_gamma_kl(t, ln_t)
        except (hypergeo.Invalid2F1, KLMinimizationFailed):
            try:
                logging.info(
                    f"Matching sufficient statistics failed with parameters: "
                    f"{a_i} {b_i} {a_j} {b_j} {y_ij} {mu_ij},"
                    f"matching mean and variance instead"
                )
                t, va_t = mutation_mean_and_variance(a_i, b_i, a_j, b_j, y_ij, mu_ij)
                proj_t = approximate_gamma_mom(t, va_t)
            except MPNoConvergence:
                raise hypergeo.Invalid2F1(
                    "Hypergeometric series does not converge; the approximate "
                    "marginal is likely degenerate.  This may reflect a topological "
                    "constraint that is at odds with the mutational data. Setting "
                    "'max_shape' to a large value (e.g. 1000) will prevent degenerate "
                    "marginals, but the results should be treated with care."
                )

    return np.array(proj_t)
