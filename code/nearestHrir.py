from pysofaconventions.SOFAFile import *
from point import *
import numpy as np
path = '/home/pi2/Documents/exjobb/tqet33-exjobb/code/mit_kemar_normal_pinna.sofa'

sofa = SOFAFile(path,'r')


def findNearestHRIR(target):
    # target = [az,el,r]
    sourcePositions = sofa.getVariableValue('SourcePosition')

    hrir = sofa.getDataIR()

    closestIdx = findNearestPositionIndex(sourcePositions,target)

    return hrir[closestIdx,:,:]


def findNearestPositionIndex(positions, target):
    N,M = np.shape(positions)
    closestDistance = np.inf
    for i in range(1,N-1):
        source = createPointFromSph(positions[i,0], positions[i,1], positions[i,2])
        distance = np.linalg.norm(np.subtract(source.cartesian(), target.cartesian()))
        if distance < closestDistance:
            closestDistance = distance
            closest = i
    return closest