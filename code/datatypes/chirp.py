import numpy as np

class Chirp:
    def __init__(self, start_freq, end_freq, length, sample_rate, amplitude):
        self.Fs = start_freq
        self.Fe = end_freq
        self.len = length
        self.sr = sample_rate
        self.A = amplitude
        self.signal = []
        
    def get_Fs(self):
        return self.Fs
    
    def get_Fe(self):
        return self.Fe
    
    def get_len(self):
        return self.len
    
    def get_amp(self):
        return self.A
    
    def get_signal(self):
        return self.signal

    def create_signal(self, scale_factor):
        length = self.len*scale_factor
        t = np.linspace(0, length, int(self.sr * length), endpoint=True)
        k = (self.Fe - self.Fs) / length  # sweep rate (Hz/s)
        #compute phase
        phase = 2 * np.pi * (self.Fs * t + 0.5 * k * t**2)
        
        self.signal = self.A * np.sin(phase)