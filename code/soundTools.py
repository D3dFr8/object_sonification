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
    #stereo = np.vstack((yL,yR)).T
    #column stack because sounddevice library requires c-contiguous memory blocks
    stereo = np.column_stack((yL,yR))

    #normalize audio and reduce volume since converting to int causes overflow
    stereo = 0.8*stereo/np.max(np.abs(stereo))

    #convert to 32 bit int for hardware limitations and multiply by max positive value
    stereo32 = np.ascontiguousarray(np.int32(stereo*2147483647))
    print(f"Sample rate: {sr}")
    sd.default.dtype = 'int32'
    sd.play(stereo32, sr)
    #sd.wait()


if __name__ == "__main__":

    y, sr = loadMP3('finger-snap.mp3')
    print(sr)
    print(y.shape)