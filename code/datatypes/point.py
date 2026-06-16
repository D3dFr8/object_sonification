import numpy as np
import datatypes.angle as angle


class Point:
    def __init__(self,x,y,z):
        self.x = x
        self.y = y
        self.z = z
        self.r = np.linalg.norm([x,y,z])
        self.az = angle.createAngleFromRadians(np.arctan2(y, x))
        self.el = angle.createAngleFromRadians(np.arcsin(z/(max(self.r,0.09))))

    def __repr__(self):
        return f"Point({self.az}, {self.el}, {self.r})"
    
    def __add__(self,other):
        return Point(self.x + other.x,
                     self.y + other.y,
                     self.z + other.z)
    
    def __sub__(self,other):
        return Point(self.x - other.x,
                     self.y - other.y,
                     self.z - other.z)
    
    def __mul__(self,other):
        if not np.isscalar(other):
            raise NotImplemented("Vector multiplication not implemented")
        scalar = float(other)
        return Point(self.x * scalar,
                     self.y * scalar,
                     self.z * scalar)

    def cartesian(self):
        return np.array([self.x,self.y,self.z])

    def spherical(self):
        return np.array([float(self.az), float(self.el), self.r])
    
    def norm(self):
        return np.sqrt(self.x**2 + self.y**2 + self.z**2)

def createPointFromSph(az,el,r):
    x = r * np.cos(float(el)) * np.cos(float(az))
    y = r * np.cos(float(el)) * np.sin(float(az))
    z = r * np.sin(float(el))
    return Point(x,y,z)

def createPointFromSphArray(target):
    return createPointFromSph(target[0],target[1],target[2])

def createPointFromCart(x,y,z):
    return Point(x,y,z)

def createPointFromCartArray(target):
    return createPointFromCart(float(target[0]),float(target[1]),float(target[2]))
