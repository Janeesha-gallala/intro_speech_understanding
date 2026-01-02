import numpy as np

def minimum_Fs(f):
    Fs = 2 * f
    return Fs

def omega(f, Fs):
    omega = 2 * np.pi * f / Fs
    return omega

def pure_tone(omega, N):
    n = np.arange(N)
    x = np.cos(omega * n)
    return x
