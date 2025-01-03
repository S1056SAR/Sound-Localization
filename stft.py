import numpy as np
import librosa
import torch
from scipy import signal
import matplotlib.pyplot as plt
import glob
def load_and_preprocess_stereo_audio(file_path, sample_rate=51200, duration=1.0):
    # Load audio file with both channels
    # mono=False ensures we keep both channels separate
    audio, orig_sr = librosa.load(file_path, sr=None, mono=False)
    if audio.shape[0] != 2:
        raise ValueError(f"Expected stereo audio but got {audio.shape[0]} channels")
    left_channel = audio[0]
    right_channel = audio[1]
    
    if orig_sr != sample_rate:
        left_channel = librosa.resample(left_channel, orig_sr=orig_sr, target_sr=sample_rate)
        right_channel = librosa.resample(right_channel, orig_sr=orig_sr, target_sr=sample_rate)
    
    target_length = int(sample_rate * duration)
    
    def adjust_length(channel, target_length):
        if len(channel) > target_length:
            return channel[:target_length]
        elif len(channel) < target_length:
            return np.pad(channel, (0, target_length - len(channel)))
        return channel
    
    left_channel = adjust_length(left_channel, target_length)
    right_channel = adjust_length(right_channel, target_length)
    
    return left_channel, right_channel

def create_stft_images(left_channel, right_channel, n_fft=2048, hop_length=512):
    
    def process_channel(channel):
        # Compute STFT
        stft = librosa.stft(channel, n_fft=n_fft, hop_length=hop_length)
        
        # Convert to magnitude spectrum
        mag_spec = np.abs(stft)
        
        # Convert to dB scale
        mag_spec_db = librosa.amplitude_to_db(mag_spec, ref=np.max)
        
        # Normalize to 0-1 range for grayscale image
        mag_spec_normalized = (mag_spec_db - mag_spec_db.min()) / (mag_spec_db.max() - mag_spec_db.min())
        
        return mag_spec_normalized
    
    left_stft = process_channel(left_channel)
    right_stft = process_channel(right_channel)
    
    return left_stft, right_stft

def visualize_stereo_preprocessing(audio_file_path):
    # Load and preprocess audio
    left_channel, right_channel = load_and_preprocess_stereo_audio(audio_file_path)
    left_stft, right_stft = create_stft_images(left_channel, right_channel)
    
    # Create visualization with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot left channel waveform
    ax1.plot(left_channel)
    ax1.set_title('Left Channel Waveform')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Amplitude')
    
    # Plot right channel waveform
    ax2.plot(right_channel)
    ax2.set_title('Right Channel Waveform')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Amplitude')
    
    # Plot left channel STFT image
    im1 = ax3.imshow(left_stft, aspect='auto', origin='lower', cmap='gray')
    ax3.set_title('Left Channel STFT')
    ax3.set_xlabel('Time Frame')
    ax3.set_ylabel('Frequency Bin')
    plt.colorbar(im1, ax=ax3)
    
    # Plot right channel STFT image
    im2 = ax4.imshow(right_stft, aspect='auto', origin='lower', cmap='gray')
    ax4.set_title('Right Channel STFT')
    ax4.set_xlabel('Time Frame')
    ax4.set_ylabel('Frequency Bin')
    plt.colorbar(im2, ax=ax4)
    
    plt.tight_layout()
    plt.show()

# Example usage
def process_stereo_dataset(file_paths):
    
    processed_left = []
    processed_right = []
    
    for file_path in file_paths:
        # Load and preprocess audio
        left_channel, right_channel = load_and_preprocess_stereo_audio(file_path)
        
        # Convert to STFT images
        left_stft, right_stft = create_stft_images(left_channel, right_channel)
        
        processed_left.append(left_stft)
        processed_right.append(right_stft)
    
    return np.array(processed_left), np.array(processed_right)

import numpy as np
import os

def save_preprocessed_data(file_paths, output_dir='preprocessed_data'):
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    for i, file_path in enumerate(file_paths):
        # Get filename without extension
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        
        # Load and preprocess audio
        left_channel, right_channel = load_and_preprocess_stereo_audio(file_path)
        left_stft, right_stft = create_stft_images(left_channel, right_channel)
        
        # Save both channels' STFT data
        np.save(
            os.path.join(output_dir, f'{base_name}_left_stft.npy'),
            left_stft
        )
        np.save(
            os.path.join(output_dir, f'{base_name}_right_stft.npy'),
            right_stft
        )
        
        # Print progress
        print(f'Processed file {i+1}/{len(file_paths)}: {base_name}')

# Usage example
audio_files = glob.glob('Audio Recordings/*.wav')
save_preprocessed_data(audio_files)
