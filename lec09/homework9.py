import numpy as np

# --------------------------------------------------
# 1) Voice Activity Detection (VAD)
# --------------------------------------------------
def VAD(waveform, Fs):
    '''
    Extract segments with energy > 10% of maximum energy.
    Frame length = 25 ms, step = 10 ms
    '''
    frame_length = int(0.025 * Fs)   # 25 ms
    step = int(0.010 * Fs)           # 10 ms

    energies = []
    frames = []

    # framing
    for start in range(0, len(waveform) - frame_length + 1, step):
        frame = waveform[start:start + frame_length]
        energy = np.sum(frame ** 2)
        energies.append(energy)
        frames.append((start, start + frame_length))

    energies = np.array(energies)
    threshold = 0.1 * np.max(energies)

    segments = []
    current_segment = []

    for i, energy in enumerate(energies):
        if energy > threshold:
            current_segment.append(frames[i])
        else:
            if len(current_segment) > 0:
                seg_start = current_segment[0][0]
                seg_end = current_segment[-1][1]
                segments.append(waveform[seg_start:seg_end])
                current_segment = []

    # catch last segment
    if len(current_segment) > 0:
        seg_start = current_segment[0][0]
        seg_end = current_segment[-1][1]
        segments.append(waveform[seg_start:seg_end])

    return segments


# --------------------------------------------------
# 2) Segments → Models
# --------------------------------------------------
def segments_to_models(segments, Fs):
    '''
    Pre-emphasize → spectrogram → low-frequency half → average log spectrum
    '''
    frame_length = int(0.004 * Fs)   # 4 ms
    step = int(0.002 * Fs)           # 2 ms
    pre_emph = 0.97

    models = []

    for seg in segments:
        # pre-emphasis
        emphasized = np.append(seg[0], seg[1:] - pre_emph * seg[:-1])

        spectra = []

        for start in range(0, len(emphasized) - frame_length + 1, step):
            frame = emphasized[start:start + frame_length]
            windowed = frame * np.hamming(frame_length)

            fft_mag = np.abs(np.fft.fft(windowed))
            low_freq = fft_mag[:len(fft_mag)//2]

            spectra.append(low_freq)

        spectra = np.array(spectra)

        # avoid log(0)
        spectra[spectra == 0] = 1e-10
        log_spectra = np.log(spectra)

        model = np.mean(log_spectra, axis=0)
        models.append(model)

    return models


# --------------------------------------------------
# 3) Speech Recognition
# --------------------------------------------------
def recognize_speech(testspeech, Fs, models, labels):
    '''
    VAD → models → cosine similarity → choose best label
    '''
    test_segments = VAD(testspeech, Fs)
    test_models = segments_to_models(test_segments, Fs)

    sims = np.zeros((len(models), len(test_models)))
    test_outputs = []

    for j, test_model in enumerate(test_models):
        for i, model in enumerate(models):
            # cosine similarity
            numerator = np.dot(model, test_model)
            denominator = np.linalg.norm(model) * np.linalg.norm(test_model)
            sims[i, j] = numerator / denominator

        best_index = np.argmax(sims[:, j])
        test_outputs.append(labels[best_index])

    return sims, test_outputs


