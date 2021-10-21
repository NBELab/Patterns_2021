import math
import numpy as np
import time
import random
import csv
from IK import *


NUMBER_OF_SAMPLES = 200
DATA_FILE = 'end_effector_test-'+str(NUMBER_OF_SAMPLES)+'.csv'
ARM_OPERATIONAL_END_EFFECTOR_ANGLE_DEGREES_RANGE = 360
#############################################################################
#
#############################################################################
def GenerateUniformDistribution(rangeX, rangeY):
    uniformArray = np.random.uniform(rangeX, rangeY, NUMBER_OF_SAMPLES)
    return uniformArray

    
#############################################################################
#
#############################################################################
eeXArray = np.empty(NUMBER_OF_SAMPLES)
eeYArray = np.empty(NUMBER_OF_SAMPLES)
eeZArray = np.empty(NUMBER_OF_SAMPLES)
isDistributionArrayEmpty = True
cosArray = np.empty(NUMBER_OF_SAMPLES)
sinArray = np.empty(NUMBER_OF_SAMPLES)
radiusArray = np.empty(NUMBER_OF_SAMPLES)

#############################################################################
# Uniform Distribution
#############################################################################
def GenerateRandomEndEffectorPositions(armPhysicalDimensions, plot=True):
    global eeXArray
    global eeYArray
    global eeZArray
    global isDistributionArrayEmpty

    #generate arrays of uniform distribution only for the first time
    if isDistributionArrayEmpty:
        eeXArray = GenerateUniformDistribution(-1, 1)
        eeYArray = GenerateUniformDistribution(0,  1) #only positive
        eeZArray = GenerateUniformDistribution(-1, 1)
        isDistributionArrayEmpty = False
    #endif

    x = eeXArray[-1]
    y = eeYArray[-1]
    z = eeZArray[-1]
    phie = math.radians(float(random.random() * ARM_OPERATIONAL_END_EFFECTOR_ANGLE_DEGREES_RANGE)) #random orientation

    eeXArray = np.delete(eeXArray, -1)
    eeYArray = np.delete(eeYArray, -1)
    eeZArray = np.delete(eeZArray, -1)

    if eeXArray.size == 0:
        isDistributionArrayEmpty = True

    if plot:
        Plot3d(eeXArray[:10000],eeYArray[:10000],eeZArray[:10000], "blue", plotNow=True) 

    return (x,y,z,phie)
#EOFunc 


#############################################################################
#
#############################################################################
def GenerateEndEffectorPositionsDatabase(arm):
    i = 0
    wrongConfig = 0
    wrongNumerical = 0
    maxNumOfNumericalIterations = 0
    db = []
    wrongDB = []
    while i < NUMBER_OF_SAMPLES:
        ePose = GenerateRandomEndEffectorPositions(arm, plot=False)
        q, trajectory, error_tract, xyz_t, result, iterations = \
            goto_target(arm, np.array([[ePose[0], ePose[1], ePose[2]]]).T, optimizer = Optimizer.STD)

        if result == False:
            wrongNumerical = wrongNumerical + 1   
            # Add data to csv
            wrongDB.append((ePose[0],ePose[1],ePose[2],    # End effector x,y,z
                      q[0][0], q[1][0], q[2][0], q[3][0], q[4][0]))   # Angles                    
        else:
            # Add data to csv
            db.append((ePose[0],ePose[1],ePose[2],    # End effector x,y,z #db.append db.insert
                       q[0][0], q[1][0], q[2][0], q[3][0], q[4][0]))   #, iterations Angles    
            i = i + 1 #increase only if legal arm configuration is found  

        if i % 100 == 0:
            print(i) #dbg

        maxNumOfNumericalIterations = maxNumOfNumericalIterations + 1      
    #while    

    print("Overall {}, wrongNumerical {} , numerical max tries {}".format(i,\
        wrongNumerical,maxNumOfNumericalIterations))

    return db,wrongDB     
    #while ends
#EOFunc

#############################################################################
#
#############################################################################
def ExportDataToCSV(db, filePrefix=""):
    with open(filePrefix+DATA_FILE,'w') as out:
        csv_out = csv.writer(out)
        for row in db:
            csv_out.writerow(row)
#EOFunc



#############################################################################
#                                    MAIN
#############################################################################
arm   = viper300()
effectorDB, wrongDB = GenerateEndEffectorPositionsDatabase(arm)
ExportDataToCSV(effectorDB)
ExportDataToCSV(wrongDB,"wrong_")   
