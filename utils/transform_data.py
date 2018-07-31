from scipy.io import wavfile as wav

from scipy.fftpack import fft

import numpy as np

rate, data = wav.read('sample-072569.wav')

fft_out = fft(data)
