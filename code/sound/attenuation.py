import numpy as np
import matplotlib.pyplot as plt

def computeAbsorptionCoefficient(f):
    h = 1
    T = 293.15
    fr0 = (24 + 4.04e4*h*(0.02+h)/(0.391+h))

    frN = 9+280*h
    A = 1
    term0 = 0.01274*(np.exp(-2239.1/T))/(fr0+(f**2/fr0))
    termN = 0.1068*(np.exp(-3352/T))/(frN+(f**2/frN))

    alpha = 8.686 * f **2 *((1.85e-11) + A * (term0 + termN))

    return alpha

def createAbsorptionFilter(alpha, r, r0):
    dr = (r - r0)*10
    H = 10 ** (-alpha * dr / 20)

    #H_full = np.concatenate([H, np.conj(H[-2:0:-1])])

    return np.fft.irfft(H)

if __name__ == "__main__":
    alpha = computeAbsorptionCoefficient(2500)
    createAbsorptionFilter(alpha, 3, 0.1)