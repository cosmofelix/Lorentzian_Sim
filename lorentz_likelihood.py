import numpy as np, pandas as pd
from matplotlib import pyplot as plt
from scipy.optimize import minimize
import scipy.stats as stats
from os import listdir

"""
###-----------------------------------###------------------------------------###
"""


def MLERegression_lorentz(params):
    nu0, A, Lambda, N = params[0], params[1], params[2], params[3]

    expect = power_expect(nu, nu0, A, Lambda)
    # negLL = -np.sum(np.log(expect + N) * np.log(y) - np.log(expect + N) ** 2)
    negLL = np.sum( np.log(expect + N) + y/(expect + N))

    return negLL


def power_expect(nu, nu0, A, Lambda):
    """expectation value of the power spectrum of l=0

    Args:
        nu:
        nu0:
        A:
        Lambda:

    Returns:

    """
    power = A * lorentz(nu, nu0, Lambda)

    return power


def lorentz(nu, nu0, Lambda):
    """Lorentzian function

    Args:
        nu (list, float): frequencies [muHz]
        Lambda (float): FWHM [muHz]
        nu0 (float): centered frequency [muHz]
    Returns:
        power (np array, float): power values [ppm muHz^-1]

    """
    power = 1 / (1 + ((nu - nu0) / (Lambda / 2)) ** 2)
    return power


def freq_err(nu, results):
    factor = (1 + results['x'][3]) ** 0.5 * ((1 + results['x'][3]) ** 0.5 + (
        results['x'][3]) ** 0.5) ** 3
    T = nu[1] - nu[0]
    return np.sqrt(factor * results['x'][2] / (4 * np.pi * T))


def uncertainties(hesse):
    term1 = 2 * hesse[0, 2] * hesse[0, 3] * hesse[3, 2]

    sigma_nu = 1/  hesse[1, 1]
    sigma_a = 1 / (hesse[0, 0] +
                   (term1 - hesse[0, 3] ** 2 * hesse
                   [2, 2] - hesse[0, 2] ** 2 * hesse[3, 3]) /
                   (hesse[2, 2] * hesse[3, 3] - hesse[2, 3] ** 2)
                   )
    sigma_gamma = 1 / (hesse[2, 2] +
                       (term1 - hesse[2, 3] ** 2 * hesse
                       [0, 0] - hesse[0, 2] ** 2 * hesse[3, 3]) /
                       (hesse[0, 0] * hesse[3, 3] - hesse[0, 3] ** 2)
                       )
    sigma_b = 1 / (hesse[3, 3] +
                   (term1 - hesse[2, 3] ** 2 * hesse
                   [0, 0] - hesse[0, 3] ** 2 * hesse[2, 2]) /
                   (hesse[0, 0] * hesse[2, 2] - hesse[0, 2] ** 2)
                   )
    return sigma_nu, sigma_a, sigma_gamma, sigma_b


def plotting(results):
    y_zero = power_expect(nu, results['x'][0], results['x'][1], results['x'][2])

    plt.figure()
    plt.plot(nu, ref, label='ref', color='black')
    plt.plot(nu, y, label='data', alpha=0.3, color='black')
    plt.plot(nu, y_zero, label='MLE')
    plt.xlabel('f / muHz')
    plt.ylabel('Power / ppm^2 muHz^-1')
    plt.legend()
    plt.show(block=False)

"""
###-----------------------------------###------------------------------------###
"""

# nu0, A, Lambda, N
guess = np.array([150, 1.5, 6, 0.02])

datapath = './batch/'

files = [f for f in listdir(datapath) if f.endswith('.dat')]
files = ['out-14.dat']
for filename in files:

    nu, ref, y, yerr = np.loadtxt(datapath+filename, unpack=True)

    results = minimize(MLERegression_lorentz, guess, method='L-BFGS-B',
                       options={'disp': True})

    print(results.x)
    # calculate uncertainties
    f_err = freq_err(nu, results)  # Libbrecht 1992
    print(f_err)
    sigma_nu, sigma_a, sigma_gamma, sigma_b = uncertainties(
        results.hess_inv.todense()) # Toutain+ 1994
    print(sigma_nu, sigma_a, sigma_gamma, sigma_b)

    plotting(results)

    if results.success:
        outstring = list(results.x)
        for i in [sigma_nu, sigma_a, sigma_gamma, sigma_b]:
            outstring.append(i)
        comments = 'MLE for ' + str(filename) + \
                   '\n'+'nu0, A, Lambda, N, nu0_err, A_err, Lambda_err, N_err'
        np.savetxt(datapath+'results-' + str(filename) + '.txt', outstring,
                   header=comments)