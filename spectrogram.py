import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from scipy.misc import imresize
from skimage.transform import resize
from numpy.lib import stride_tricks
import torch
import pandas as pd

""" short time fourier transform of audio signal """
def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)   
    samples = np.append(np.zeros(int(np.floor(frameSize/2.0))), sig)    
    # cols for windowing
    cols = np.ceil( (len(samples) - frameSize) / float(hopSize)) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(int(cols), frameSize), strides=(samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)    

""" scale frequency axis logarithmically """    
def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale))

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):        
        if i == len(scale)-1:
            newspec[:,i] = np.sum(spec[:,int(scale[i]):], axis=1)
        else:        
            newspec[:,i] = np.sum(spec[:,int(scale[i]):int(scale[i+1])], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[int(scale[i]):])]
        else:
            freqs += [np.mean(allfreqs[int(scale[i]):int(scale[i+1])])]

    return newspec, freqs

def get_stft_ims(audiopath, binsize=2**12):
    samplerate, samples = wav.read(audiopath)

    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)

    ims = 20.*np.log10(np.abs(sshow)/10e-6) # amplitude to decibel
    
    ims = ims[1:,:] ## chop off -inf artifact

    ims = resize(ims,(100,512)) ## resize spectrogram to have constant bins
    ims = ims/np.nanmax(ims) ## normalize amplitiude

    return ims

def plot_ims(ims, plotpath=None, colormap="jet"):
    """ plot spectrogram"""

    timebins, freqbins = np.shape(ims)
    
    plt.figure(figsize=(15, 7.5))
    plt.imshow(np.transpose(ims), origin="lower",
               aspect="auto", cmap=colormap, interpolation="none")
    colorbar = plt.colorbar()
    colorbar.set_label("Power (arbitrary)")
    plt.xlabel("Time (arbitrary)")
    plt.ylabel("Frequency (arbitrary)")
    
    plt.xlim([0, timebins])
    plt.ylim([0, freqbins])

    xlocs = np.linspace(0, timebins, 5)
    plt.xticks(xlocs, xlocs/timebins)
    
    ylocs = np.linspace(0, freqbins, 5)
    plt.yticks(ylocs, ylocs/freqbins)
    
    
    if plotpath:
        plt.savefig(plotpath, bbox_inches="tight")
    else:
        plt.show()

    plt.clf()

    return ims

def make_plots(df, n_samples=10, prefix='', path=''):
    """
    save n_samples of spectrograms from dataframe
    """
    for wav_file in df.filename.sample(n_samples):
        ims = get_stft_ims(path+wav_file)
        plot_ims(ims,'figs/'+prefix+wav_file.replace('.wav','.png').split('/')[1])

def generate_stft_data(df, n_train=1000, n_test=100):
    """
    save n_samples of spectrograms as pytorch data from dataframe
    """

    ## empty list to append with data
    train_data = []
    train_targets = []
    test_data = []
    test_targets = []

    for ix, row in df.sample(n_train+n_test).iterrows():
        wav_file = row.filename
        ims = get_stft_ims(path+wav_file)
        target = int(row.gender=='female')
        if ix<n_train:
            train_data.append(ims)
            train_targets.append(target)
        else:
            test_data.append(ims)
            test_targets.append(target)
            
    train_data = np.array(train_data)
    train_targets = np.array(train_targets)
    test_data = np.array(test_data)
    test_targets = np.array(test_targets)

    train_data = torch.tensor(train_data, dtype=torch.float32)
    train_targets = torch.tensor(train_targets, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    test_targets = torch.tensor(test_targets, dtype=torch.float32)

    torch.save((train_data,train_targets),open('train.pt','wb'))
    torch.save((test_data,test_targets),open('test.pt','wb'))
        
        
if __name__  == '__main__':

    ## location of uncompressed data
    path = '/home/troydsvm/common_voice/cv_corpus_v1/'

    df = pd.read_csv(path+'cv-valid-test.csv')

    df.filename = df.filename.str.replace('.mp3','.wav')
    
    male = df[df.gender=='male']
    female = df[df.gender=='female']

    #make_plots(male,prefix='male-',path=path)
    #make_plots(female,prefix='female-',path=path)

    df = pd.concat([male,female])

    
    generate_stft_data(df)
