import numpy as np
from point import *


def projectPointOnSphere(p, radius, earPoint):
    dV = p - earPoint
    a = np.dot(dV.cartesian(),dV.cartesian())
    b = 2*np.dot(earPoint.cartesian(), dV.cartesian())
    c = np.dot(earPoint.cartesian(), earPoint.cartesian()) - radius**2

    t = np.roots([a,b,c])

    newPoint = earPoint + dV*max(t)

    return newPoint

def interpolationWeights(points, target):
    weights = np.zeros_like(points)
    for i in range(0,np.size(points)):
        distance = (points[i] - target).norm()
        if (distance == 0):
            weights = np.zeros_like(points)
            weights[i] = 1
            return weights
        weights[i] = 1 / distance

    weights = weights / np.sum(weights)
    return weights

def closestPointIdx(points, target, N):
    distances = [(i, (p - target).norm()**2)
                 for i, p in enumerate(points)]
    
    distances.sort(key=lambda x:x[1])
    return [i for i, d in distances[:N]]