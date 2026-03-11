import sound.loadHrir as loadHrir
import sound.interpolateHrir as interpolateHrir
import torch
import matplotlib.pyplot as plt
import numpy as np
import sound.attenuation as attenuation
import sound.soundTools as soundTools


hrir = loadHrir.getImpulseResponses()
N = len(hrir[1,1,:])
sourcePositions = loadHrir.getSourcePositions()[:,:2]

#path = 'mit_kemar_net.pth'
#net,X_mean,X_std = loadHrir.loadModel(path)

earPoints = np.array(loadHrir.getEarPositions())
sourcePoints = np.array(loadHrir.getSourcePoints())

fs = loadHrir.getSamplingRate()

def getHrirAtTargetNN(targetPoint):
    """
    Docstring for getHrirAtTargetNN
    
    :param targetPoint: Is of the class Point

    :returns hL, hR: Are the interpolated head related impulse responses using NN
    """

    pointL = interpolateHrir.projectPointOnSphere(targetPoint, sourcePoints[0].spherical()[2], earPoints[0])
    pointR = interpolateHrir.projectPointOnSphere(targetPoint, sourcePoints[0].spherical()[2], earPoints[1])

    tensorL = torch.tensor(pointL.spherical()[:2], dtype=torch.float32)
    tensorLNorm = (tensorL - X_mean) / X_std
    predicted_hrirL = net(tensorLNorm).detach().numpy()[0,1,:]

    tensorR = torch.tensor(pointR.spherical()[:2], dtype=torch.float32)
    tensorRNorm = (tensorR - X_mean) / X_std
    predicted_hrirR = net(tensorRNorm).detach().numpy()[0,0,:]

    return predicted_hrirL, predicted_hrirR

f = np.arange(0, N//2 + 1) * (fs / N)

alpha = attenuation.computeAbsorptionCoefficient(f)

def getHrirAtTarget(targetPoint, r0):

    """
    Docstring for getHrirAtTarget
    
    :param targetPoint: Is of the class Point

    :returns hL, hR: Are the interpolated head related impulse responses using inverted distance
    """

    pointL = interpolateHrir.projectPointOnSphere(targetPoint, sourcePoints[0].spherical()[2], earPoints[0])
    pointR = interpolateHrir.projectPointOnSphere(targetPoint, sourcePoints[0].spherical()[2], earPoints[1])

    idxL = interpolateHrir.closestPointIdx(sourcePoints, pointL, 3)
    idxR = interpolateHrir.closestPointIdx(sourcePoints, pointR, 3)

    wL = interpolateHrir.interpolationWeights(sourcePoints[idxL], pointL)
    wR = interpolateHrir.interpolationWeights(sourcePoints[idxR], pointR)

    hLs = np.array(hrir[idxL,1,:]) * wL[:,None] 
    hRs = np.array(hrir[idxR,0,:]) * wR[:,None]

    hL = np.array(np.sum(hLs, axis=0))
    hR = np.array(np.sum(hRs, axis=0))

    hAtt = np.abs(attenuation.createAbsorptionFilter(alpha, targetPoint.r, r0))

    hL = soundTools.conv(hL,hAtt)
    hR = soundTools.conv(hR,hAtt)

    return hL, hR

#if __name__ == "__main__":

 #   getHrirAtTarget(point.createPointFromSphArray([0,0,1]))