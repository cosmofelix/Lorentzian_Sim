#  Copyright (c) 2020. Felix Mackebrandt

from matplotlib import pyplot as plt
import numpy as np
import math
from scipy import special as sp
import pandas as pd
import sys

"""
###-----------------------------------###------------------------------------###
"""


def lorentz(nu, nu0, Lambda, m, Omega):
    """Lorentzian function

    Args:
        nu (list, float): frequencies [muHz]
        Lambda (float): FWHM [muHz]
        nu0 (float): centered frequency [muHz]
        m (float): azimuthal degree
        Omega (float): angular velocity of the star
    Returns:
        power (np array, float): power values [ppm muHz^-1]

    Note:
    damping rate omega = pi * Lambda
    """
    power = 1 / (1 + (((nu - m * Omega) - nu0) / (Lambda / 2)) ** 2)
    return power


def e_lm(l, m, i):
    """dependence of mode power on azimuthal order m

    Args:
        l:
        m:
        i:

    Returns:

    """
    if l == 0:
        return 1
    if l == 1:
        if  m == 0:
            return np.cos(i)**2
        if np.abs(m) == 1:
            return 0.5 * np.sin(i)**2
        else:
            print('l or m error')
            sys.exit()
    if l == 2:
        if  m == 0:
            return 0.25 * (3 * np.cos(i)**2 -1)**2
        if np.abs(m) == 1:
            return 0.375 * np.sin(2 * i)**2
        if np.abs(m) == 2:
            return 0.375 * np.sin(i)**4
        else:
            print('l or m error')
            sys.exit()
    else:
        print('l or m error')
        sys.exit()


def uniform_dist(omega):
    return np.random.uniform(0, 1, omega)


def power_expect(l, m, incl, nu, nu0, A, Lambda, Omega):
    """expectation value of the power spectrum

    Args:
        l:
        m:
        i:
        nu:
        nu0:
        A:
        Lambda:
        Omega:

    Returns:

    """
    power = 0
    for i in m:
        power += e_lm(l, i, incl) * lorentz(nu, nu0, Lambda, i, Omega)
    power = A * power

    return power


def background(nu, A, tau):
    """background model

    for two power laws (solar background) after Harvey (1985)

    Args:
        nu (list, float): frequencies [muHz]
        A (list, float): amplitude (two elements) [ppm^2 muHz^-1]
        tau (list, float): time scale (two elements) [s]
    Returns:
        power (np array, float): power values [ppm^2 muHz^-1]
    """
    tau = np.divide(tau, 10 ** 6)  # [s] in [mus]
    power = A[0] / (1 + (tau[0] * nu) ** 2) + A[1] / (1 + (tau[1] * nu) ** 2)
    return power


def noise(signal, sn=500):
    """Gaussian (white) noise

    Args
        signal (list, float): points
        sn (float):	signal to noise ratio
    Returns
        flux (np array, float): signal + noise ()
        flux_err (np array, float): uncertainty of 'flux'
    """
    flux = np.random.normal(signal, 1 / sn)
    flux_err = 1 / sn * np.sqrt(signal)

    return flux, flux_err


def multiplet():
    # TODO
    return 0


def plotting_power(data):
    """plotting the power spectrum

    Args:
        data (pandas df): frequencies 'nu' [muHz], 'power' and 'power_err [ppm^2
        muHz^-1]
    """
    data.plot('nu', ['power', 'model'])
    plt.xlabel('f / muHz')
    plt.ylabel('Power / ppm^2 muHz^-1')
    plt.legend()
    plt.show(block=False)

    return 0


def saving(data, filename, info):
    comments = 'omega, l , Omega, incl, nu0, A_lor, Lambda \n' + str(info) + \
               '\n' + str(data.columns.values.tolist())

    np.savetxt(filename, data, header=comments)


"""
###-----------------------------------###------------------------------------###
"""

# model input parameter
omega = 500 # angular frequency sample 2 pi j / T
l = 0
m = [i for i in range(-l,l+1,1)] # azimuthal order, [-l,...,l]
Omega = 0 # muHz, angular velocity of the star
incl = 0 # 0.555

# Lorentz function
nu0 = 150 # muHz
A_lor = 1 # ppm
Lambda = 2 * np.pi # muHz

# background function
A_back = [1.6, 0.5]
tau = [1400, 500]


data = pd.DataFrame(np.linspace(nu0 - 50, nu0 + 50, num=omega), columns=[
    'nu'])

for i in np.arange(1000):
    model = power_expect(l, m, incl, data['nu'], nu0, A_lor, Lambda, Omega)

    power_realization, power_realization_err = noise(model)
    power_realization = np.multiply(power_realization,
                                    -np.log(uniform_dist(omega)))

    data = data.assign(model=model, power=power_realization,
                       power_err=power_realization_err)

    saving(data, 'batch/out-'+str(i)+'.dat', [omega, l, Omega, incl, nu0,
                                              A_lor, Lambda])


plotting_power(data)