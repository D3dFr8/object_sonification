import librosa
import numpy as np
import sounddevice as sd
import asyncio

def loadMP3(filename, sr=None):
    y, sr = librosa.load(filename, sr=sr, mono=True)
    return y, sr

def resample(y, sr_old, sr_new):
    n_samples = y.shape[-1]
    n_new = int(np.round(n_samples * sr_new / sr_old))
    old_idx = np.arange(n_samples)
    new_idx = np.linspace(0,n_samples-1, n_new)

    return np.interp(new_idx, old_idx, y)

def conv(A,B):
    return np.convolve(A,B)

def playSound(yL,yR,sr):
    stereo = np.vstack((yL,yR)).T
    sd.play(stereo, sr)
    sd.wait()


if __name__ == "__main__":

    y, sr = loadMP3('finger-snap.mp3')
    print(sr)
    print(y.shape)