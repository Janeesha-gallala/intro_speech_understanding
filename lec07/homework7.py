import numpy as np

def major_chord(f, Fs):
    '''
    Generate a one-half-second major chord, based at frequency f,
    with sampling frequency Fs.
    '''
    N = int(0.5 * Fs)
    n = np.arange(N)

    # Major chord frequencies
    f1 = f
    f2 = f * (2 ** (4/12))
    f3 = f * (2 ** (7/12))

    x = (
        np.cos(2 * np.pi * f1 * n / Fs) +
        np.cos(2 * np.pi * f2 * n / Fs) +
        np.cos(2 * np.pi * f3 * n / Fs)
    )
    return x


def dft_matrix(N):
    '''
    Create a DFT transform matrix of size N.
    '''
    n = np.arange(N)
    k = n.reshape((N, 1))
    W = np.cos(2 * np.pi * k * n / N) - 1j * np.sin(2 * np.pi * k * n / N)
    return W


def spectral_analysis(x, Fs):
    '''
    Find the three loudest frequencies in x.
    '''
    N = len(x)

    W = dft_matrix(N)
    X = np.dot(W, x)
    mag = np.abs(X)

    # Keep only positive frequencies (avoid mirrored peaks)
    mag = mag[:N // 2]

    # Remove DC component
    mag[0] = 0

    # Find three strongest peaks
    idx = np.argsort(mag)[-3:]
    freqs = idx * Fs / N

    freqs = np.sort(freqs)
    return freqs[0], freqs[1], freqs[2]
