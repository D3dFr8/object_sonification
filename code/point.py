import numpy as np


class Point:
    x = 0
    y = 0
    z = 0
    az = 0
    el = 0
    r = 0

    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
        self.r = np.linalg.norm([x,y,z])
        self.az = np.arccos(z/(self.r+0.0001))
        self.el = np.arctan(y / (x+0.0001))

    def cartesian(self):
        return np.array([self.x,self.y,self.z])

    def spherical(self):
        return np.array([self.az, self.el, self.r])

def createPointFromSph(az,el,r):
    x = r * np.cos(el) * np.cos(az)
    y = r * np.cos(el) * np.sin(az)
    z = r * np.sin(el)
    return Point(x,y,z)

def createPointFromCart(x,y,z):
    return Point(x,y,z)
