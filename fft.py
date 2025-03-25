import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import SpanSelector
from scipy.io import wavfile

class InteractiveWAVAnalyzer:
    def __init__(self, file_path, num_harmonics=5):
        # Load audio file
        self.sample_rate, self.audio_data = wavfile.read(file_path)
        
        # If stereo, convert to mono by averaging channels
        if len(self.audio_data.shape) > 1:
            self.audio_data = np.mean(self.audio_data, axis=1)
        
        # Total duration of the audio
        self.total_duration = len(self.audio_data) / self.sample_rate
        
        # Setup the plot
        self.fig, (self.ax1, self.ax2, self.ax3, self.ax4) = plt.subplots(4, 1, figsize=(12, 10))
        plt.subplots_adjust(hspace=0.4)
        
        # Plot full waveform
        self.time_axis = np.linspace(0, self.total_duration, len(self.audio_data))
        self.ax1.plot(self.time_axis, self.audio_data)
        self.ax1.set_title("Full Audio Waveform")
        self.ax1.set_xlabel("Time (seconds)")
        self.ax1.set_ylabel("Amplitude")
        
        # Create span selector for selecting segment
        self.span = SpanSelector(
            self.ax1, 
            self.on_select, 
            'horizontal', 
            useblit=True,
            button=[1],  # Left mouse button
            minspan=0.001  # Minimum selection span
        )
        
        # Prepare subplot placeholders
        self.harmonics_line = None
        self.reconstructed_line = None
        self.original_segment_line = None
        self.spectrum_line = None
        
        plt.show()
    
    def on_select(self, xmin, xmax):
        # Clear previous plots
        for ax in [self.ax2, self.ax3, self.ax4]:
            ax.clear()
        
        # Extract the selected segment
        start_sample = int(xmin * self.sample_rate)
        end_sample = int(xmax * self.sample_rate)
        
        # Ensure we don't go out of bounds
        start_sample = max(0, start_sample)
        end_sample = min(len(self.audio_data), end_sample)
        
        selected_audio = self.audio_data[start_sample:end_sample]
        
        # Time axis for selected segment
        segment_duration = len(selected_audio) / self.sample_rate
        segment_time_axis = np.linspace(xmin, xmax, len(selected_audio))
        
        # Compute FFT
        fft_result = np.fft.fft(selected_audio)
        freqs = np.fft.fftfreq(len(selected_audio), 1 / self.sample_rate)
        
        # Get magnitudes
        magnitudes = np.abs(fft_result[:len(freqs)//2])
        
        # Find dominant frequencies (top 5)
        num_harmonics = 5
        dominant_indices = np.argsort(magnitudes)[-num_harmonics:]
        dominant_freqs = freqs[dominant_indices]
        dominant_amps = magnitudes[dominant_indices] / np.max(magnitudes)
        
        # Reconstruct harmonic components
        reconstructed_signal = np.zeros_like(selected_audio, dtype=float)
        harmonic_components = []
        
        # Plot individual harmonic components
        self.ax2.clear()
        for i in range(num_harmonics):
            harmonic = dominant_amps[i] * np.sin(2 * np.pi * dominant_freqs[i] * segment_time_axis)
            harmonic_components.append(harmonic)
            self.ax2.plot(segment_time_axis, harmonic, label=f"Harmonic {i+1}: {dominant_freqs[i]:.2f} Hz")
            reconstructed_signal += harmonic
        
        self.ax2.set_title("Individual Harmonic Components")
        self.ax2.set_xlabel("Time (seconds)")
        self.ax2.set_ylabel("Amplitude")
        self.ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        
        # Plot reconstructed harmonic signal
        self.ax3.clear()
        self.ax3.plot(segment_time_axis, reconstructed_signal, color='green')
        self.ax3.set_title(f"Harmonic Reconstruction ({num_harmonics} Components)")
        self.ax3.set_xlabel("Time (seconds)")
        self.ax3.set_ylabel("Amplitude")
        
        # Plot selected audio segment
        self.ax4.clear()
        self.ax4.plot(segment_time_axis, selected_audio, color='blue')
        self.ax4.set_title("Selected Audio Segment")
        self.ax4.set_xlabel("Time (seconds)")
        self.ax4.set_ylabel("Amplitude")
        
        # Update frequency spectrum
        plt.subplot(4, 1, 3)
        plt.plot(freqs[:len(freqs)//2], magnitudes, color='red')
        plt.xlim(0, 10000)
        plt.title("Frequency Spectrum")
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Magnitude")
        
        # Redraw the figure
        plt.tight_layout()
        self.fig.canvas.draw_idle()

# Usage
analyzer = InteractiveWAVAnalyzer("test.wav", 10)
