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
    #column stack because sounddevice library requires c-contiguous memory blocks
    stereo = np.column_stack((yL,yR))

    #normalize audio and reduce volume since converting to int can cause overflow
    #stereo = stereo/np.max(np.abs(stereo))

    #set as C-contiguous array for the sake of the hardware
    stereo32 = np.ascontiguousarray(stereo, dtype=np.float32)

    try:
        sd.play(stereo32, sr)
        sd.wait()
    except Exception as e:
        print(f"Audio playing failed. Exception occured: {e}")


if __name__ == "__main__":

    y, sr = loadMP3('finger-snap.mp3')
    print(sr)
    print(y.shape)