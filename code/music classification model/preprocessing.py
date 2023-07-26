import librosa , librosa.display
import matplotlib.pyplot as plt
import numpy as np

file = "blues.00000.wav"

# waveform
# loading the audio file , also specify sample rate
# sr x T = > 22050 x 30 
signal , sr = librosa.load(file,sr=22050)

# visualize wavefor,
librosa.display.waveshow(signal,sr=sr)
# specify axis label
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()


# moving from time domain to frequency domain
# fft transform
# fft => spectrum
fft = np.fft.fft(signal)

magnitude = np.abs(fft)

# linspace -> number of evenly spaced numbers in an interval
frequency = np.linspace(0,sr,len(magnitude))

# plt.plot(frequency,magnitude)

# # specify axis label
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()

left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(frequency)/2)]

# plt.plot(left_frequency,left_magnitude)

# # specify axis label
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()


# stft -> spectogram
n_fft = 2048
hop_length = 512

stft = librosa.core.stft(signal,hop_length=hop_length,n_fft=n_fft)

spectogram = np.abs(stft)

log_spectogram = librosa.amplitude_to_db(spectogram)
# librosa.display.specshow(log_spectogram,sr=sr,hop_length=hop_length)

# # specify axes label
# plt.xlabel("Time")
# plt.ylabel("Magnitude")
# plt.show()

# MFCC

MFCCs = librosa.feature.mfcc(signal,n_fft=n_fft,hop_length=hop_length,n_mfcc=13)
librosa.display.specshow(MFCCs,sr=sr,hop_length=hop_length)

# specify axes label
plt.xlabel("Time")
plt.ylabel("MFCCs")
plt.show()