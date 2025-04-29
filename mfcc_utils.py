import numpy as np
import scipy.fftpack
import scipy.signal
import librosa

def load_audio(file_path, sr=16000):
    """Load the audio file and return the signal and sampling rate."""
    signal, sample_rate = librosa.load(file_path, sr=sr)
    return signal, sample_rate

def frame_signal(signal, frame_size, frame_stride, sample_rate):
    """Slice the signal into overlapping frames."""
    frame_length = int(round(frame_size * sample_rate))
    frame_step = int(round(frame_stride * sample_rate))
    signal_length = len(signal)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) + 1

    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(signal, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    return frames

def apply_hamming_window(frames):
    """Apply Hamming window to each frame."""
    return frames * np.hamming(frames.shape[1])

def compute_fft_power(frames, NFFT=512):
    """Compute power spectrum of each frame."""
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = ((1.0 / NFFT) * (mag_frames ** 2))
    return pow_frames

def mel_filterbank(sample_rate, NFFT, nfilt=26):
    """Generate Mel filterbank."""
    low_freq_mel = 0
    high_freq_mel = 2595 * np.log10(1 + (sample_rate / 2) / 700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = 700 * (10**(mel_points / 2595) - 1)
    bin = np.floor((NFFT + 1) * hz_points / sample_rate).astype(int)

    fbank = np.zeros((nfilt, int(NFFT / 2 + 1)))
    for m in range(1, nfilt + 1):
        f_m_minus = bin[m - 1]   # left
        f_m = bin[m]             # center
        f_m_plus = bin[m + 1]    # right

        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

    return fbank

def compute_mfcc(pow_frames, fbank, num_ceps=13):
    """Apply Mel filter bank, log compression, and DCT to extract MFCCs."""
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    log_fbanks = np.log(filter_banks)
    mfcc = scipy.fftpack.dct(log_fbanks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
    return mfcc

def extract_mfcc(signal, sample_rate, frame_size=0.025, frame_stride=0.01, num_ceps=13, nfilt=26, NFFT=512):
    """Complete MFCC pipeline."""
    frames = frame_signal(signal, frame_size, frame_stride, sample_rate)
    windowed = apply_hamming_window(frames)
    power_spectrum = compute_fft_power(windowed, NFFT)
    fbank = mel_filterbank(sample_rate, NFFT, nfilt)
    mfcc = compute_mfcc(power_spectrum, fbank, num_ceps)
    return mfcc, fbank, power_spectrum, windowed, frames