from pysofaconventions import *
from point import *
import angle
import numpy as np
import PointToIrNN
import torch
path = 'mit_kemar_normal_pinna.sofa'

sofa = SOFAFile(path,'r')

def getSourcePositions():
    positions = sofa.getVariableValue('SourcePosition')
    positions_angle = positions.copy()
    
    positions_angle[:,0] = [angle.createAngleFromDegrees(positions[i,0])
                           for i in range(positions_angle.shape[0])]
    positions_angle[:,1] = [angle.createAngleFromDegrees(positions[i,1])
                           for i in range(positions_angle.shape[0])]

    return positions_angle

def getSourcePoints():
    sourcePositions = getSourcePositions()
    return [
        createPointFromSphArray(sourcePositions[i,:])
        for i in range(sourcePositions.shape[0])
    ]

def getEarPositions():
    earPositions = sofa.getVariableValue("ReceiverPosition")
    return [
        createPointFromCartArray(earPositions[i,:])
        for i in range(earPositions.shape[0])
    ]

def getSamplingRate():
    return sofa.getSamplingRate()[0]

def getImpulseResponses():
    return sofa.getDataIR()

def loadModel(path):
    file = torch.load(path)
    net = PointToIrNN.PointToIrNN()
    net.load_state_dict(file['model_state_dict'])
    net.eval()

    X_mean = file['X_mean']
    X_std = file['X_std']

    return net, X_mean, X_std