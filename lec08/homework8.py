
import numpy as np

def waveform_to_frames(waveform, frame_length, step):
    N = len(waveform)
    num_frames = 1 + (N - frame_length) // step

    frames = np.zeros((num_frames, frame_length))

    for i in range(num_frames):
        start = i * step
        frames[i, :] = waveform[start:start + frame_length]

    return frames


def frames_to_mstft(frames):
    num_frames, frame_length = frames.shape
    mstft = np.zeros((num_frames, frame_length))

    for i in range(num_frames):
        X = np.fft.fft(frames[i])
        mstft[i] = np.abs(X)

    return mstft


def mstft_to_spectrogram(mstft):
    max_val = np.amax(mstft)
    threshold = max(0.001 * max_val, 1e-12)

    mstft = np.maximum(mstft, threshold)
    spectrogram = 20 * np.log10(mstft)

    max_db = np.amax(spectrogram)
    spectrogram = np.maximum(spectrogram, max_db - 60)

    return spectrogram
