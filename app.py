import streamlit as st
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from mfcc_utils import load_audio, extract_mfcc

def plot_waveform(signal, sample_rate):
    """Display the waveform of the signal."""
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(signal, sr=sample_rate)
    plt.title("Time-domain Signal")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    st.pyplot()

def plot_spectrogram(power_spectrum, sample_rate, NFFT=512):
    """Display power spectrum of the signal."""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(power_spectrum, ref=np.max),
                             sr=sample_rate, x_axis='time', y_axis='log', hop_length=NFFT//2)
    plt.title("Power Spectrum")
    plt.colorbar(format="%+2.0f dB")
    st.pyplot()

def plot_mel_filters(fbank, sample_rate, NFFT=512):
    """Display the mel filter bank."""
    plt.figure(figsize=(10, 4))
    plt.imshow(fbank, aspect='auto', origin='lower')
    plt.title("Mel Filter Bank")
    plt.ylabel("Mel Filter Index")
    plt.xlabel("Frequency (Hz)")
    st.pyplot()

def plot_mfcc(mfcc):
    """Display the MFCCs as a heatmap."""
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc.T, x_axis='time', sr=16000)
    plt.title("MFCCs")
    plt.ylabel("MFCC Coefficients")
    plt.colorbar()
    st.pyplot()

# Streamlit App Interface
st.title("MFCC-based Speaker Recognition")

st.sidebar.header("Upload your audio file:")
uploaded_file = st.sidebar.file_uploader("Choose a .wav file", type=["wav"])

if uploaded_file is not None:
    # Load audio file
    signal, sample_rate = load_audio(uploaded_file)

    # Display the waveform
    st.subheader("Time-domain Signal:")
    plot_waveform(signal, sample_rate)

    # Process MFCCs
    mfcc, fbank, power_spectrum, windowed, frames = extract_mfcc(signal, sample_rate)

    # Display Spectrogram
    st.subheader("Power Spectrum:")
    plot_spectrogram(power_spectrum, sample_rate)

    # Display Mel Filter Bank
    st.subheader("Mel Filter Bank:")
    plot_mel_filters(fbank, sample_rate)

    # Display MFCCs
    st.subheader("MFCCs (Heatmap):")
    plot_mfcc(mfcc)
    
    # Display additional information
    st.sidebar.info("Submitted by: Aayush Kanungu | Roll No: 2311401132")