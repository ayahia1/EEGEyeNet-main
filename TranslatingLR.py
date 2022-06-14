import numpy as np
import math

def main():
    dmin = np.load('data/Direction_task_with_dots_synchronised_min_hilbert.npz')

    ldmin = dmin['labels']

    cdmin = leftrighttrans1(ldmin[:,2])

    ndmin = np.vstack((ldmin[:,0], cdmin)).T

    np.savez('./data/LR_task_with_dots_synchronised_min_hilbert.npz', EEG = dmin['EEG'], labels = ndmin)

def leftrighttrans1(rarr):
    narr = []
    cnt1 = 0
    cnt2 = 0
    for i in range(len(rarr)):
        if abs(rarr[i]) < math.pi/2:
            narr.append(1)
            cnt1+=1
        else:
            narr.append(0)
            cnt2+=1
    print("Right: " + str(cnt1))
    print("Left: " + str(cnt2))
    return narr

main()