import numpy as np
from scipy.stats import vonmises
from scipy.integrate import quad


def compute_von_mises_coverage(
        dPsi, sigma, weights=None, quantiles=np.linspace(0.001, 1, 100)):
    """Compute Coverage based on von Mises-Fisher Distribution

    Parameters
    ----------
    dPsi : array_like
        The opening angle between true and reconstructed direction.
    sigma : array_like
        The (circularized) uncertainty estimate for each event.
    weights : array_like, optional
        The event weights.
    quantiles : array_like, optional
        The quantile values for which to compute the coverage

    Returns
    -------
    array_like
        The quantile values at which the coverage is computed.
    array_like
        The coverage values for each of the quantiles
    """
    cdf_values = von_mises_in_dPsi_cdf(dPsi, sigma)
    return compute_coverage(cdf_values, weights=weights, quantiles=quantiles)


def compute_coverage(
        cdf_values,
        weights=None,
        quantiles=np.linspace(0.001, 1, 100)):
    """Compute Coverage

    Parameters
    ----------
    cdf_values : array_like
        The cumulative distribution function evaluated
        at each (dir, unc) pair.
    weights : array_like, optional
        The event weights
    quantiles : array_like, optional
        The quantile values for which to compute the coverage

    Returns
    -------
    array_like
        The quantile values at which the coverage is computed.
    array_like
        The coverage values for each of the quantiles
    """
    if weights is None:
        weights = np.ones_like(cdf_values)

    num_events = np.sum(weights)

    # sort values
    sorted_indices = np.argsort(cdf_values)
    sorted_cdf_values = cdf_values[sorted_indices]
    sorted_weights = weights[sorted_indices]

    cum_sum = np.cumsum(sorted_weights) / num_events
    indices = np.searchsorted(sorted_cdf_values, quantiles)
    mask_over = indices >= len(cum_sum)
    if np.sum(mask_over) > 0:
        print('Clipping {} events to max.'.format(np.sum(mask_over)))
        indices = np.clip(indices, 0, len(cum_sum) - 1)
    coverage = cum_sum[indices]

    return quantiles, coverage


def scipy_von_mises_pdf(x, sigma):
    kappa = 1. / sigma**2
    return vonmises.pdf(x, kappa=kappa)


def von_mises_pdf(x, sigma, kent_min=np.deg2rad(7)):
    """PDF in (dx1, dx2), i.e x = sqrt(dx1^2 + dx2^2)
    """
    x = np.atleast_1d(x)
    sigma = np.atleast_1d(sigma)

    cos_dpsi = np.cos(x)
    kappa = 1. / sigma**2
    result = np.where(
        kent_min < sigma,
        (
            # kappa / (4 * np.pi * np.sinh(kappa)) *
            # np.exp(kappa * cos_dpsi)

            # stabilized version:
            kappa / (4 * np.pi)
            * 2 * np.exp(kappa * (cos_dpsi - 1.)) / (1. - np.exp(-2.*kappa))
        ),
        # 1./(2*np.pi*sigma**2) * np.exp(-x**2 / (2*sigma**2)),
        1./(2*np.pi*sigma**2) * np.exp(-0.5 * (x / sigma)**2),
    )
    return result


def von_mises_in_dPsi_pdf(x, sigma):
    """PDF in dPsi
    """
    # switching coordinates from (dx1, dx2) to spherical
    # coordinates (dPsi, phi) means that we have to include
    # the jakobi determinant sin dPsi
    jakobi_det = np.sin(x)  # sin(dPsi)
    phi_integration = 2 * np.pi
    return phi_integration*jakobi_det * von_mises_pdf(x, sigma)


def von_mises_in_dPsi_cdf(x, sigma):
    """CDF in dPsi
    """
    x = np.atleast_1d(x)
    sigma = np.atleast_1d(sigma)
    result = []
    for x_i, sigma_i in zip(x, sigma):
        integration = quad(
            von_mises_in_dPsi_pdf, a=0, b=x_i, args=(sigma_i,))
        result.append(integration[0])
    return np.array(result)


def gauss2d(x, sigma):
    """PDF in (dx1, dx2), i.e x = sqrt(dx1^2 + dx2^2)
    """
    return 1./(2*np.pi*sigma**2) * np.exp(-x**2 / (2*sigma**2))


def gauss2d_dsigma(x, sigma):
    """d/dsigma gauss_2d

    PDF in (dx1, dx2), i.e x = sqrt(dx1^2 + dx2^2)
    """
    exp = np.exp(-x**2 / (2*sigma**2))
    nominator = x**4 - 7*x**2*sigma**2 + 6*sigma**4
    denominator = 2 * np.pi * sigma**8
    return nominator / denominator * exp


def gauss2d_d2sigma(x, sigma):
    """d^2/dsigma^2 gauss_2d

    PDF in (dx1, dx2), i.e x = sqrt(dx1^2 + dx2^2)
    """
    exp = np.exp(-x**2 / (2*sigma**2))
    nominator = x**2 - 2*sigma**2
    denominator = 2 * np.pi * sigma**5
    return nominator / denominator * exp


def rayleigh(x, sigma):
    """PDF in dPsi
    """
    return (x/(sigma**2)) * np.exp(- (x**2 / (2 * sigma**2)))


def rayleigh_cdf(x, sigma):
    """CDF in dPsi
    """
    return 1 - np.exp(- (x**2 / (2 * sigma**2)))
