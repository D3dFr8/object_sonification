import numpy as np

class Angle:
    def __init__(self,v):
        self.rad = v # Radians by default
        self.deg = v*180/np.pi

    def radians(self):
        return self.rad
    
    def degrees(self):
        return self.deg
    
    def __float__(self):
        return self.radians()
    
def createAngleFromDegrees(v):
    return Angle(v*np.pi/180)

def createAngleFromRadians(v):
    return Angle(v)
