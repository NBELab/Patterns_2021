import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import csv
import math
import random
from datetime import datetime
from packaging import version
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import sklearn
from tensorflow import keras
from keras import backend as BK
from keras.layers import Dense
from keras.callbacks import TensorBoard
from numpy import loadtxt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error
from timeit import default_timer as timer
from keras import layers, models
import sys,threading
import io
import nengo
import nengo_dl
from IK_tensor import *

# Supply here h5 format trained model file to be converted to SNN
##################################################################################
MODEL_NAME = '../../Data/fc-4hl-relu.h5'
##################################################################################
ENABLE_LOG = 1
VERBOSE = 1

DATA_FILE = '../../Data/end_effector_test-200000.csv'

LOAD_MODEL = 1 # Change to 1 to skip training, load existing model and predict
DO_NENGO = 1

NUM_OF_TEST_SAMPLES = 30000
START_LAYER = 4
END_LAYER = 5
EPOCHS = 2000
BATCH_SIZE = 1
NORMALIZE_DATA=False
TRAIN_WITH_KERAS=True

KRS_HLAYER_ACT_FUNC=tf.nn.relu
KRS_OLAYER_ACT_FUNC=tf.keras.activations.linear 

NUM_OF_INPUT_VARIABLES = 3
NUM_OF_OUTPUT_VARIABLES = 5
NUM_OF_NEURONS_PER_HLAYER = 256
NENGO_STEPS = 250

############################################################################
#
############################################################################
sc =  MinMaxScaler() # (0,1)
globalAllDataNonNorm = []
maxXYZ = 0.0
minXYZ = 0.0
maxAngles = 0.0
minAngles = 0.0
arm = viper300()


def mystr(mystr, mystr2=None):
    if mystr2:
        return "_"+mystr2+"_"+str(mystr)
    else:
        return "_"+str(mystr)


##################################################################################
isEager = tf.executing_eagerly() #dbg

DEBUG_PRINT_ON = 1
def DebugPrint(str):
    if DEBUG_PRINT_ON:
        print(str)


#############################################################################
#
#############################################################################
def Normalize(data, fitOnly=False):
    global sc
    if fitOnly:
        normalizedData = sc.fit(data)
    else:
       normalizedData = sc.transform(data)
    return normalizedData

#############################################################################
#
#############################################################################
def DeNormalize(data):
    global sc
    deNormalizedData = sc.inverse_transform(data)
    return deNormalizedData

############################################################################
#
############################################################################
def LoadCsvData(fileName):
    dataset = loadtxt(fileName, delimiter=',')

    allDataXyzNonNorm = dataset[:,0:3]
    allDataAnglesNonNorm = dataset[:,3:8]

    return dataset, allDataXyzNonNorm, allDataAnglesNonNorm
#end    

#############################################################################
#
#############################################################################
def LoadData(normalize = False, limitDataToXSamples=None):
    global sc
    global arm
    global maxXYZ
    global minXYZ
    global maxAngles
    global minAngles

    # load the dataset
    dataset = loadtxt(DATA_FILE, delimiter=',')

    if limitDataToXSamples:
        dataset = dataset[:limitDataToXSamples,0:8]

    NUM_OF_SAMPLES = len(dataset)
    NUM_OF_TRAINING_SAMPLES = int(NUM_OF_SAMPLES * 0.7)
    DATA_VALIDATION_PERCENT=0.15
    NUM_OF_VALIDATION_SAMPLES = int(NUM_OF_SAMPLES * DATA_VALIDATION_PERCENT)

    allDataXyzNonNorm = dataset[:,0:3]
    allDataAnglesNonNorm = dataset[:,3:8]

    # !!!!! Shift all angles to be positive and then between 0-2pi !!!!!
    dataset[:,3:8] = dataset[:,3:8] + 100*np.pi
    dataset[:,3:8] = dataset[:,3:8] % (2*np.pi)

    # split into input and output variables
    endEffectorData = allDataXyzNonNorm[0:NUM_OF_TRAINING_SAMPLES] #effector x,y,z
    anglesData = allDataAnglesNonNorm[0:NUM_OF_TRAINING_SAMPLES]      #theta1, 2, 3, 4, 5 in RAD

    endEffectorDataValidation = allDataXyzNonNorm[NUM_OF_TRAINING_SAMPLES:NUM_OF_TRAINING_SAMPLES + \
                                        NUM_OF_VALIDATION_SAMPLES] #effector x,y,z
    anglesDataValidation = allDataAnglesNonNorm[NUM_OF_TRAINING_SAMPLES:NUM_OF_TRAINING_SAMPLES + \
                                   NUM_OF_VALIDATION_SAMPLES]      #theta1, 2, 3, 4, 5 in RAD

    endEffectorDataTest = allDataXyzNonNorm[NUM_OF_TRAINING_SAMPLES+NUM_OF_VALIDATION_SAMPLES:] #effector x,y, z
    anglesDataTest = allDataAnglesNonNorm[NUM_OF_TRAINING_SAMPLES+NUM_OF_VALIDATION_SAMPLES:]   #theta1, 2, 3, 4, 5 in RAD

    if normalize:     #Normalizing the data
        maxXYZ = np.amax(allDataXyzNonNorm)
        minXYZ = np.amin(allDataXyzNonNorm)
        maxAngles = np.amax(allDataAnglesNonNorm)
        minAngles = np.amin(allDataAnglesNonNorm)

        allDataXyzNonNormFlat = np.reshape(allDataXyzNonNorm,(-1,1)) #1 single column all
        sc.fit(allDataXyzNonNormFlat) #fit xyz
        #sc.fit(allDataXyzNonNorm) #fit xyz
        endEffectorDataNormalized = sc.transform(endEffectorData) #! fit
        endEffectorDataValidationNormalized = sc.transform(endEffectorDataValidation)
        endEffectorDataTestNormalized = sc.transform(endEffectorDataTest) #only transform

        allDataAnglesNonNormFlat = np.reshape(allDataAnglesNonNorm,(-1,1))    #1 single column all
        sc.fit(allDataAnglesNonNormFlat) #fit angles
        #sc.fit(allDataAnglesNonNorm) #fit xyz
        anglesDataNormalized = sc.transform(anglesData)
        anglesDataValidationNormalized = sc.transform(anglesDataValidation)
        anglesDataTestNormalized = sc.transform(anglesDataTest)

        # Scaling validity check:
        unscaleAnglesTest1 = np.array([tensorInverseTransform(i) for i in anglesDataTestNormalized])

        comparison = np.allclose(unscaleAnglesTest1, anglesDataTest, rtol=1e-013, atol=1e-013) 
        if not comparison:
            print("\n!!!! Bug in scaling/descaling mechanism !!!!\n")


        allDataAnglesNonNormFlat = np.reshape(allDataAnglesNonNorm,(-1,1))    #1 single column all
        Normalize(allDataAnglesNonNormFlat, True) #fit only for scaler
        unscaleAnglesTest2  = DeNormalize(anglesDataTestNormalized)

        comparison = np.allclose(unscaleAnglesTest2, anglesDataTest, rtol=1e-13, atol=1e-013) #unscaleAnglesTest2 == anglesDataTest; areArraysEqual = comparison.all()
        if not comparison:
            print("\n!!!! Bug in scaling/descaling mechanism !!!!\n")

        #shift angles to be 0-2pi and then Mod 2pi
        tansformedAnglesData = allDataAnglesNonNorm + 100*np.pi
        tansformedAnglesData = tansformedAnglesData % (2*np.pi)

        PredictAndCalcFkMse(arm, dataset, allDataXyzNonNorm, allDataAnglesNonNorm, normalize=False, plotFile="onload_") #dbg
        PredictAndCalcFkMse(arm, dataset, allDataXyzNonNorm, tansformedAnglesData, normalize=False, plotFile="onload_") #dbg
        PredictAndCalcFkMse(arm, dataset, endEffectorData, anglesDataNormalized, normalize=True, plotFile="onload_") #dbg
########      
    else:
        endEffectorDataNormalized = endEffectorData
        endEffectorDataValidationNormalized = endEffectorDataValidation
        endEffectorDataTestNormalized = endEffectorDataTest

        anglesDataNormalized = anglesData
        anglesDataValidationNormalized = anglesDataValidation
        anglesDataTestNormalized = anglesDataTest

    return endEffectorDataNormalized, anglesDataNormalized, \
           endEffectorDataValidationNormalized, anglesDataValidationNormalized, \
           endEffectorDataTestNormalized, anglesDataTestNormalized, \
           endEffectorDataTest, anglesDataTest, \
           dataset #not normalized


#############################################################################
#
#############################################################################
def RunNetwork(
    model,
    inputTest,
    outputTest,
    kerasInputTest=[],
    activation=None,
    params_file=MODEL_NAME,
    n_steps=NENGO_STEPS,
    scale_firing_rates=1,
    synapse=None,
    n_test=NUM_OF_TEST_SAMPLES,
    showPlot=False,
    hlayers=2,
    useKerasModel=False,
    loadKerasModelFromFile=False,
    savePlotToFile=True
):

    # convert the keras model to a nengo network
    if useKerasModel:
        if loadKerasModelFromFile:
            model.load_weights(params_file,by_name=True)

            #add Bias to move angles between 0-2pi
            model.build((None, 3))
            model._layers[-1].bias.assign_add(np.array(2*np.pi).repeat(5))           

            if len(kerasInputTest) > 0:
                print("\nRunning pure KERAS predict...",len(kerasInputTest)," samples")
                startClock = timer()  
                predictTest = model.predict(kerasInputTest)
                endClock = timer()
                print("Average inference time: ",(endClock-startClock)*1000/(len(kerasInputTest)), ' ms/sample\n')              


    if activation:
        nengo_converter = nengo_dl.Converter(
            model,
            swap_activations={KRS_HLAYER_ACT_FUNC: activation, KRS_OLAYER_ACT_FUNC:None },
            scale_firing_rates=scale_firing_rates,
            synapse=synapse,
        )
    else:
        nengo_converter = nengo_dl.Converter(
            model,
            scale_firing_rates=scale_firing_rates,
            synapse=synapse,
        )      

    inputLayer = model.get_layer('input_1')
    hiddenLayer1 = model.get_layer('dense') #first hidden layer
    outputLayer = model.get_layer("dense_" + str(hlayers)) #output layer

    # get input/output objects
    nengo_input = nengo_converter.inputs[inputLayer]
    nengo_output = nengo_converter.outputs[outputLayer]

    # add a probe to the first layer to record activity, record from a subset of neurons, to save memory.
    sample_neurons = np.linspace(
        0,
        np.prod(hiddenLayer1.output_shape[1:]),
        n_steps,
        endpoint=False,
        dtype=np.int32,
    )
    with nengo_converter.net:
        probe_l1 = nengo.Probe(nengo_converter.layers[hiddenLayer1][sample_neurons])

    # repeat inputs for some number of timesteps
    testInputData = np.tile(inputTest[:n_test], (1, n_steps, 1))

    # set some options to speed up simulation
    with nengo_converter.net:
        nengo_dl.configure_settings(planner=nengo_dl.graph_optimizer.noop_planner) # ! Disable nengo optimizer for load model
        nengo_dl.configure_settings(stateful=False)        

    # build network, load in trained weights, run inference on test images
    with nengo_dl.Simulator(
        nengo_converter.net, minibatch_size=BATCH_SIZE, progress_bar=False,
    ) as nengo_sim:
        if not useKerasModel:
            print("Loading nengo params...")
            nengo_sim.load_params(params_file)
        startClock = timer()    
        data = nengo_sim.predict({nengo_input: testInputData})
        endClock = timer()
        print("\nAverage inference time: ",(endClock-startClock)*1000/(len(inputTest)), ' ms/sample')

    mse = MSE(outputTest[:n_test], data[nengo_output][:n_test, -1]) #normalized comparison
    print('MSE of ANGLES Nengo run IK net prediction of {} samples: {}'.format(len(inputTest), mse))

    plt.close()             
    sys.stdout.flush()

    return data[nengo_output][:n_test, -1]


############################################################################
#
#############################################################################
def euclidean_distance_loss(truth, pred):
    tt = tf.convert_to_tensor(truth, dtype=tf.float64)
    pt = tf.convert_to_tensor(pred, dtype=tf.float64)
    return BK.sqrt(BK.sum(BK.square(pt - tt), axis=-1))

############################################################################
#
#############################################################################
#@tf.function
def MSE(truth, pred):
    a = mean_squared_error(truth, pred)
    return a

#############################################################################
#
#############################################################################
def PredictAndCalcFkMse(arm, allDataNonNorm, truthEePos, predAngles, normalize=True, plotFile="plot.png", showPlot=False):
    allDataXyzNonNorm = allDataNonNorm[:,0:3]
    allDataAnglesNonNorm = allDataNonNorm[:,3:8]
    anglesDataTest = allDataAnglesNonNorm[-NUM_OF_TEST_SAMPLES:,:] #dbg
    eeDataTest = allDataXyzNonNorm[-NUM_OF_TEST_SAMPLES:,:] #dbg

    if normalize:
        allDataAnglesNonNormFlat = np.reshape(allDataAnglesNonNorm,(-1,1))    #1 single column all
        Normalize(allDataAnglesNonNormFlat, True) #fit only for scaler
        DeNormalizedAngles = DeNormalize(predAngles)
    else:
        DeNormalizedAngles = predAngles

    eePredictedPos = []
    for i in DeNormalizedAngles:
        eeXYZ = FK(arm, np.array(i))
        eePos = (eeXYZ[0][0], eeXYZ[1][0], eeXYZ[2][0]) #xyz np.array to list
        eePredictedPos.append(np.array(eePos))
    #for        

    eePredictedPosNpArr = np.array(eePredictedPos)

    mse = MSE(truthEePos, eePredictedPosNpArr) #non normalized comparison
    eloss = euclidean_distance_loss(truthEePos, eePredictedPosNpArr) #returns a tensor

    proto = tf.make_tensor_proto(eloss)
    npEloss = tf.make_ndarray(proto) #returns np array from tensor
    largerThan1Cm = npEloss[npEloss > 0.01]
    largerThan1CmIndices = np.nonzero(npEloss > 0.01)
    largerThan1CmArrayPred = eePredictedPosNpArr[largerThan1CmIndices,]
    largerThan1CmArrayPred = largerThan1CmArrayPred[0,:,:]

    largerThan1Mm = npEloss[npEloss > 0.001]
    largerThan1MmIndices = np.nonzero(npEloss > 0.001)
    largerThan1MmArrayPred = eePredictedPosNpArr[largerThan1MmIndices,]
    largerThan1MmArrayPred = largerThan1MmArrayPred[0,:,:]

    largerThan1CmArrayTruth = truthEePos[largerThan1CmIndices,]
    largerThan1CmArrayTruth = largerThan1CmArrayTruth[0,:,:]

    print('MSE prediction of {} samples for FK: {}\n'.format(np.size(predAngles,0), mse))
    print('Average Euclidean Distance to target in Meters: {}, min distance: {}, \
         max distance: {} Points above 1 cm distance: {} ; Points above 1 mm distance: {}\n'.format(BK.mean(eloss, axis=0), 
         BK.min(eloss, axis=0), BK.max(eloss, axis=0), np.size(largerThan1CmArrayPred,0),np.size(largerThan1MmArrayPred,0) ))

    sys.stdout.flush()         
#end

#############################################################################
#
#############################################################################
def ReturnModel(arm,inputTrain,outputTrain,inputValidation,outputValidation,inputTest,outputTest, hlayers=2):

    steps = 1
    inputTrainNgo = np.tile(inputTrain[:, None, :], (1, steps, 1))
    outputTrainNgo = np.tile(outputTrain[:, None, :], (1, steps, 1))
    inputValidationNgo = np.tile(inputValidation[:, None, :], (1, steps, 1))
    outputValidationNgo = np.tile(outputValidation[:, None, :], (1, steps, 1))
    inputTestNgo = np.tile(inputTest[:, None, :], (1, steps, 1))
    outputTestNgo = np.tile(outputTest[:, None, :], (1, steps, 1))

    BK.clear_session()
    netLayers = []
    input = tf.keras.Input(shape=(NUM_OF_INPUT_VARIABLES,))
    l1 = tf.keras.layers.Dense(NUM_OF_NEURONS_PER_HLAYER, activation=KRS_HLAYER_ACT_FUNC)(input)
    netLayers.append(l1)
    for layer in range(1, hlayers):
        tl = tf.keras.layers.Dense(NUM_OF_NEURONS_PER_HLAYER, activation=KRS_HLAYER_ACT_FUNC)(netLayers[layer-1])
        netLayers.append(tl)
    output = tf.keras.layers.Dense(NUM_OF_OUTPUT_VARIABLES, activation=KRS_OLAYER_ACT_FUNC)(netLayers[-1])

    model = tf.keras.Model(inputs=input, outputs=output)

    model.summary()

    sys.stdout.flush()

    return model, inputTestNgo
#end


#############################################################################
# Main
#############################################################################
if ENABLE_LOG:
    sys.stdout = open(MODEL_NAME+mystr(LOAD_MODEL)+mystr(datetime.now().strftime("%Y%m%d-%H%M%S"))+"-log.txt", "w")

print(tf.keras.__version__)
if tf.test.gpu_device_name():
    print("\nGPU found\n")
else:
    print("\nNo GPU found\n")

print (MODEL_NAME)

if LOAD_MODEL:
    inputTrain,outputTrain,inputValidation,outputValidation, inputTest, \
        outputTest, eeTestDataNonNorm, eeTestAnglesNonNorm, allDataNonNorm = LoadData(NORMALIZE_DATA)

    allDataXyzNonNorm = allDataNonNorm[:,0:3]
    globalAllDataNonNorm = allDataNonNorm

    for l in range(START_LAYER, END_LAYER):
        NNa, inputTestNgo = ReturnModel(arm, inputTrain, outputTrain, inputValidation,
            outputValidation, inputTest, outputTest, hlayers=l)   


        print("\n####################### Regular Net #######################\n")
        sys.stdout.flush()
        MF = MODEL_NAME

        netPrediction = RunNetwork(
            NNa,
            inputTestNgo,
            outputTest,
            kerasInputTest=inputTest,
            activation=nengo.RectifiedLinear(),
            hlayers=l,
            params_file=MF,
            useKerasModel=TRAIN_WITH_KERAS,
            showPlot=False,
            savePlotToFile=True,
            loadKerasModelFromFile=True,
            n_steps=1
        )        
        #note net prediction was done on a scaled input and returns a scaled output
        PredictAndCalcFkMse(arm, allDataNonNorm, eeTestDataNonNorm, netPrediction, showPlot=False,
                normalize=NORMALIZE_DATA, plotFile=MF)

        if DO_NENGO:
            syn = [0.011, 0.018, 0.02, 0.022, 0.025, 0.03]
            fr =  [5000, 1500, 1000, 750, 500, 250]
            for f in (fr):
                for i in syn:
                    print("\n####################### Spiking Net #######################\n", i, f)
                    netPrediction = RunNetwork(
                        NNa,
                        inputTestNgo,
                        outputTest,
                        activation=nengo.SpikingRectifiedLinear(),
                        n_steps = NENGO_STEPS,
                        synapse = i,
                        scale_firing_rates=f,
                        hlayers=l,
                        params_file=MF,
                        useKerasModel=TRAIN_WITH_KERAS,
                        loadKerasModelFromFile=True,
                        showPlot=False,
                        savePlotToFile=True,
                    )

                    PredictAndCalcFkMse(arm, allDataNonNorm, eeTestDataNonNorm, netPrediction, showPlot=False,
                            normalize=NORMALIZE_DATA, plotFile=MF+mystr(syn)+mystr(f))
                    sys.stdout.flush()
                #end for
            #end for
        #endif
    #end for    

sys.stdout.close()
