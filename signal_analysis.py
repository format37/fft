import numpy as np
import matplotlib.pyplot as plt

def generate_signal(sinusoids, duration=1.0, sample_rate=200, noise_level=0.0):
    """Generate combined signal from list of [amplitude, frequency] pairs with optional noise"""
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    signal = np.zeros_like(t)
    
    for amplitude, frequency in sinusoids:
        signal += amplitude * np.cos(2 * np.pi * frequency * t)
    
    signal = signal / len(sinusoids)
    
    # Add noise if specified
    if noise_level > 0:
        noise = np.random.normal(0, noise_level, len(signal))
        signal += noise
    
    return t, signal

def fft_analysis(signal, sample_rate):
    """Perform FFT analysis and extract frequencies and amplitudes"""
    # Compute FFT
    fft = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(len(signal), 1/sample_rate)
    
    # Take only positive frequencies
    positive_freq_mask = frequencies >= 0
    frequencies = frequencies[positive_freq_mask]
    fft_magnitude = np.abs(fft[positive_freq_mask])
    
    # Find peaks (frequencies with significant amplitude)
    # Use threshold to filter out noise
    threshold = 0.1 * np.max(fft_magnitude)
    peak_indices = np.where(fft_magnitude > threshold)[0]
    
    detected_freqs = frequencies[peak_indices]
    detected_amplitudes = 2 * fft_magnitude[peak_indices] / len(signal)  # Scale back to original amplitude
    
    return detected_freqs, detected_amplitudes

def reconstruct_sinusoids(detected_freqs, detected_amplitudes, t):
    """Convert detected frequencies and amplitudes back to sinusoids"""
    reconstructed_signals = []
    
    for freq, amp in zip(detected_freqs, detected_amplitudes):
        if freq > 0:  # Skip DC component
            sinusoid = amp * np.cos(2 * np.pi * freq * t)
            reconstructed_signals.append((freq, amp, sinusoid))
    
    return reconstructed_signals

def plot_analysis(t, original_signal, reconstructed_signals, detected_freqs, detected_amplitudes, original_sinusoids):
    """Plot dedicated comparisons: individual sinusoids and summed signals"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Individual sinusoids comparison
    # Generate original individual sinusoids (without noise, without normalization)
    for i, (amp, freq) in enumerate(original_sinusoids):
        original_sinusoid = amp * np.cos(2 * np.pi * freq * t)
        ax1.plot(t, original_sinusoid, 'b-', linewidth=1, alpha=0.7, 
                label=f'Original: {amp} * cos(2π * {freq} * t)' if i < 4 else "")
    
    # Plot restored sinusoids (only main frequencies, skip noise artifacts)
    main_freqs = [8, 10, 25, 35]  # Expected main frequencies
    for freq, amp, sinusoid in reconstructed_signals:
        if freq in main_freqs:
            # Scale back up by 4 to match original amplitude
            scaled_sinusoid = sinusoid * 4
            ax1.plot(t, scaled_sinusoid, 'r--', linewidth=1, alpha=0.8,
                    label=f'Restored: {amp*4:.1f} * cos(2π * {freq:.0f} * t)')
    
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Original vs Restored Individual Sinusoids')
    ax1.grid(True, alpha=0.3)
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.set_xlim(0, 0.3)
    
    # Plot 2: Summed signals comparison
    # Original clean signal (without noise)
    original_clean_signal = np.zeros_like(t)
    for amplitude, frequency in original_sinusoids:
        original_clean_signal += amplitude * np.cos(2 * np.pi * frequency * t)
    original_clean_signal = original_clean_signal / len(original_sinusoids)
    
    # Restored summed signal (sum of main reconstructed components)
    restored_summed_signal = np.zeros_like(t)
    for freq, amp, sinusoid in reconstructed_signals:
        if freq in main_freqs:
            restored_summed_signal += sinusoid
    
    ax2.plot(t, original_clean_signal, 'b-', linewidth=2, label='Original Clean Signal')
    ax2.plot(t, restored_summed_signal, 'r--', linewidth=2, label='Restored Summed Signal')
    
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title('Original vs Restored Summed Signals')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(0, 0.5)
    
    # Plot 3: Noisy signal vs all reconstructed components
    ax3.plot(t, original_signal, 'b-', linewidth=1, alpha=0.8, label='Noisy Signal')
    for i, (freq, amp, sinusoid) in enumerate(reconstructed_signals[:6]):  # Show first 6 components
        if freq > 0:
            ax3.plot(t, sinusoid, '--', linewidth=1, alpha=0.6,
                    label=f'{amp:.2f} * cos(2π * {freq:.0f} * t)')
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Noisy Signal vs All Reconstructed Components')
    ax3.grid(True, alpha=0.3)
    ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.set_xlim(0, 0.3)
    
    # Plot 4: Frequency spectrum
    ax4.stem(detected_freqs, detected_amplitudes, basefmt=' ')
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('Amplitude')
    ax4.set_title('Detected Frequencies and Amplitudes')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 50)
    
    plt.tight_layout()
    plt.savefig('/home/alex/projects/fft/fft_analysis.png', dpi=300, bbox_inches='tight')
    # plt.show()  # Comment out to avoid blocking

def main():
    # Define sinusoids as [amplitude, frequency] pairs
    sinusoids = [
        [2.0, 10],  # 2.0 amplitude, 10 Hz
        [1.5, 25],  # 1.5 amplitude, 25 Hz
        [3.0, 8],   # 3.0 amplitude, 8 Hz
        [1.0, 35]   # 1.0 amplitude, 35 Hz
    ]
    
    # Parameters
    duration = 1.0
    sample_rate = 200
    
    # Generate signal with significant noise
    noise_level = 0.3  # Add significant noise
    t, signal = generate_signal(sinusoids, duration, sample_rate, noise_level)
    
    # Perform FFT analysis
    detected_freqs, detected_amplitudes = fft_analysis(signal, sample_rate)
    
    # Reconstruct sinusoids
    reconstructed_signals = reconstruct_sinusoids(detected_freqs, detected_amplitudes, t)
    
    # Print results
    print("Original sinusoids:")
    for i, (amp, freq) in enumerate(sinusoids):
        print(f"  {i+1}: {amp} * cos(2π * {freq} * t)")
    
    print(f"\nNoise level: {noise_level}")
    
    print(f"\nDetected frequencies and amplitudes:")
    for i, (freq, amp) in enumerate(zip(detected_freqs, detected_amplitudes)):
        if freq > 0:  # Skip DC component
            print(f"  {i+1}: {amp:.2f} * cos(2π * {freq:.1f} * t)")
    
    # Plot analysis
    plot_analysis(t, signal, reconstructed_signals, detected_freqs, detected_amplitudes, sinusoids)

if __name__ == "__main__":
    main()