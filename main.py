import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

def plot_waveform_and_spectrum(file_path, num_harmonics=5):
    # Load audio file
    sample_rate, audio_data = wavfile.read(file_path)
    
    # If stereo, convert to mono by averaging channels
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)

    start = 2
    start_time = 0.001 + start
    end_time = 0.005 + start

    # Convert to sample indices
    start_sample = int(start_time * sample_rate)
    end_sample = int(end_time * sample_rate)

    # Extract the segment
    audio_data = audio_data[start_sample:end_sample]
    
    # Time axis for waveform
    duration = len(audio_data) / sample_rate
    time_axis = np.linspace(0, duration, len(audio_data))
    
    # Compute FFT
    fft_result = np.fft.fft(audio_data)
    freqs = np.fft.fftfreq(len(audio_data), 1 / sample_rate)
    
    # Get dominant frequencies
    magnitudes = np.abs(fft_result[:len(freqs)//2])
    dominant_indices = np.argsort(magnitudes)[-num_harmonics:]
    dominant_freqs = freqs[dominant_indices]
    dominant_amps = magnitudes[dominant_indices] / np.max(magnitudes)
    
    # Reconstruct harmonic components
    reconstructed_signal = np.zeros_like(audio_data, dtype=float)
    harmonic_components = []
    
    plt.figure(figsize=(12, 8))
    
    # Plot individual harmonic components
    plt.subplot(4, 1, 1)
    for i in range(num_harmonics):
        harmonic = dominant_amps[i] * np.sin(2 * np.pi * dominant_freqs[i] * time_axis * sample_rate)
        harmonic_components.append(harmonic)
        plt.plot(time_axis, harmonic, label=f"Harmonic {i+1}: {dominant_freqs[i]:.2f} Hz")
        reconstructed_signal += harmonic
    plt.title("Individual Harmonic Components")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    plt.legend()
    
    # Plot reconstructed harmonic signal
    plt.subplot(4, 1, 2)
    plt.plot(time_axis, reconstructed_signal, color='green')
    plt.title(f"Harmonic Reconstruction ({num_harmonics} Components)")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    
    # Plot original waveform
    plt.subplot(4, 1, 3)
    plt.plot(time_axis, audio_data, color='blue')
    plt.title("Original Audio Waveform")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude")
    
    # Plot frequency spectrum
    plt.subplot(4, 1, 4)
    plt.plot(freqs[:len(freqs)//2], magnitudes, color='red')
    plt.xlim(0, 10000)
    plt.title("Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Magnitude")
    
    plt.tight_layout()
    plt.show()

plot_waveform_and_spectrum("test.wav", num_harmonics=5)

