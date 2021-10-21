import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #-1 = Disable GPU

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from utils import RelativeReduceLROnPlateau, RelativeEarlyStopping, EarlyStoppingWMinEpoch, FKc
from scipy.spatial import ckdtree
from typing import NamedTuple
import datetime
import sys
import csv
from skspatial.objects import Line
from skspatial.objects import Sphere
from IK import *

PREDICT_ONLY = False
EPOCHS = 80000
PAIR_DISTANCE_IN_M = 0.02

OBSTACLE_RADIUS_IN_M = 0.1
PENALTY_DISTANCE_EPSILON = 2.5
SUM_OF_ARM_JOINTS_LEN_MM = 985

class Obstacle(NamedTuple): #sphere defined by its center coordinates and radius
    x: float
    y: float
    z: float
    r: float
#class

o1 = Obstacle(0, 0.3, 0, OBSTACLE_RADIUS_IN_M)
o2 = Obstacle(0.15, 0.2, 0.15, OBSTACLE_RADIUS_IN_M)
o3 = Obstacle(0, 0.45, 0, OBSTACLE_RADIUS_IN_M)
o4 = Obstacle(-0.5, 0.07, -0.5, OBSTACLE_RADIUS_IN_M)
o5 = Obstacle(0.35, 0.6, -0.35, OBSTACLE_RADIUS_IN_M)
ObstacleList = [o1] #Train for 1 obstacle, for multiple obstacles use example: [o1,o2,o3,o4,o5]


OBSTACLE_RADIUS_SQUARED_IN_M = OBSTACLE_RADIUS_IN_M ** 2

DATA_FILE =  "../../Data/end_effector_test-200000.csv"

def test_NN_IK_configuration(num_layers, num_ftrs, activation, use_angle_norm, batch_size):

    num_layers = num_layers
    num_ftrs = num_ftrs
    activation_name = ['lerelu', 'relu', 'tanh', 'swish', 'mish'][activation]
    activation = [tf.nn.leaky_relu, tf.nn.relu, tf.nn.tanh,  tf.keras.activations.swish, tfa.activations.mish][activation]
    angle_norm=float(use_angle_norm)*1.e-5
    norm_type = use_angle_norm
    batch_size = batch_size
    f=1

    name = 'fc200kenergy'+str(PAIR_DISTANCE_IN_M*100)+'cm_obstcl_'+ datetime.datetime.now().strftime("%Y%m%d-%H%M%S")+'_{num_layers}_{num_ftrs}_{activ}_{theta_norm}_{bs}_{ob_num}_{ob_rad}_{pend}_lrsched2'.\
        format(num_layers=num_layers,num_ftrs=num_ftrs, activ=activation_name, 
               theta_norm=norm_type, bs=batch_size, ob_num=len(ObstacleList), ob_rad=OBSTACLE_RADIUS_IN_M, pend=PENALTY_DISTANCE_EPSILON)
    lr_init = 1e-1
    num_epochs = EPOCHS

    fkc = FKc(batch_size*2)

    #################################################################################
    #################################################################################

    def fk_mean_dist(gt, pred):
        pred_end_effector_pos = fkc.FK(pred)[:,-1]
        gt_end_effector_pos = tf.reshape(gt, (batch_size*2, 3))
        dist = tf.sqrt(tf.reduce_sum(tf.square(pred_end_effector_pos-gt_end_effector_pos), axis=1))
        return tf.reduce_mean(dist)

    def fk_std_dist(gt, pred):
        pred_end_effector_pos = fkc.FK(pred)[:,-1]
        gt_end_effector_pos = tf.reshape(gt, (batch_size*2, 3))
        dist = tf.sqrt(tf.reduce_sum(tf.square(pred_end_effector_pos-gt_end_effector_pos), axis=1))
        mean, var = tf.nn.moments(dist, axes=0)
        return tf.sqrt(var)

    def fk_max_dist(gt, pred):
        pred_end_effector_pos = fkc.FK(pred)[:,-1]
        gt_end_effector_pos = fkc.FK(gt)[:,-1]
        return tf.reduce_max(tf.sqrt(tf.reduce_sum(tf.square(pred_end_effector_pos-gt_end_effector_pos), axis=1)))

    def fk_median_dist(gt, pred):
        pred_end_effector_pos = fkc.FK(pred)[:,-1]
        gt_end_effector_pos = fkc.FK(gt)[:,-1]
        dist = tf.sqrt(tf.reduce_sum(tf.square(pred_end_effector_pos-gt_end_effector_pos), axis=1))
        return tf.nn.top_k(dist, batch_size//2+1).values[-1]

    def fk_above_1cm_dist(gt, pred):
        pred_end_effector_pos = fkc.FK(pred)[:,-1]
        gt_end_effector_pos = tf.reshape(gt, (batch_size*2, 3))
        dist = tf.sqrt(tf.reduce_sum(tf.square(pred_end_effector_pos-gt_end_effector_pos), axis=1))
        above_1cm = tf.reduce_sum(tf.cast(dist>1.e-2, tf.float32))
        return above_1cm/(batch_size*2)*100

    def fk_above_1mm_dist(gt, pred):
        pred_end_effector_pos = fkc.FK(pred)[:,-1]
        gt_end_effector_pos = tf.reshape(gt, (batch_size*2, 3))
        dist = tf.sqrt(tf.reduce_sum(tf.square(pred_end_effector_pos-gt_end_effector_pos), axis=1))
        above_1mm = tf.reduce_sum(tf.cast(dist>1.e-3, tf.float32))
        return above_1mm/(batch_size*2)*100

    def fk_inside_obstacle(gt, pred):
        var = -1.5
        d = 0.0
        joints = fkc.FK(pred)

        for o in ObstacleList:
            ocenter = [o.x, o.y, o.z]
            o_r_sqr = o.r**2

            for i in range(0, 6):
                dist_sqr1 = tf.reduce_sum(tf.square(joints[:,i] - ocenter), axis=1)  
                collide1 = tf.reduce_sum(tf.cast(dist_sqr1 <= o_r_sqr, tf.float32))
                d = d + collide1

            for i in range(1, 6):
                midpoint_joint_pos = (joints[:,i] + joints[:,i-1]) / 2
                dist_sqr2 = tf.reduce_sum(tf.square(midpoint_joint_pos - ocenter), axis=1)   
                collide2 = tf.reduce_sum(tf.cast(dist_sqr2 <= o_r_sqr, tf.float32))
                d = d + collide2

        #for obstacles
        
        return (d) / (batch_size*2)*100       
    #end    
    #################################################################################

    #################################################################################
    def close_pairs_ckdtree(points, max_d):
        tree = ckdtree.cKDTree(points)
        pairs = tree.query_pairs(max_d)
        return np.array(list(pairs))

    #####      Data handling      #####
    data = np.loadtxt(DATA_FILE, delimiter=',').astype(np.float32)
    cartesian_coord, angles = data[:, :3], data[:, 3:] #duplicate
    angles = np.fmod(angles, 2*np.pi)
    angles[angles>=np.pi] -= 2*np.pi
    angles[angles<-np.pi] += 2*np.pi

    dataset = tf.data.Dataset.from_tensor_slices((cartesian_coord, cartesian_coord))
    dataset = dataset.shuffle(buffer_size=int(1e3),seed=0).batch(batch_size, drop_remainder=True)

    num_train = round(len(dataset)*0.7)
    num_valid = round(len(dataset)*0.15)

    train = dataset.take(num_train)

    np_train_x = np.empty((len(train),batch_size, 3))
    for i, x in enumerate(train.as_numpy_iterator()):
        np_train_x[i] = x[0].copy()
    np_train_x = np_train_x.reshape((-1, 3))

    inds = close_pairs_ckdtree(np_train_x, PAIR_DISTANCE_IN_M)

    coords = np_train_x[inds]
    train = tf.data.Dataset.from_tensor_slices((coords, coords)).batch(batch_size, drop_remainder=True)

    val = dataset.skip(num_train).take(num_valid)
    np_val_x = np.empty((len(val),batch_size, 3))
    for i, x in enumerate(val.as_numpy_iterator()):
        np_val_x[i] = x[0].copy()
    np_val_x = np_val_x.reshape((-1, 3))
    coords = np_val_x.reshape((-1,2,3))
    val = tf.data.Dataset.from_tensor_slices((coords, coords)).batch(batch_size, drop_remainder=True)

    test = dataset.skip(num_train+num_valid)
    np_test_x = np.empty((len(test),batch_size, 3))
    for i, x in enumerate(test.as_numpy_iterator()):
        np_test_x[i] = x[0].copy()
    np_test_x = np_test_x.reshape((-1, 3))
    coords = np_test_x.reshape((-1,2,3))
    test = tf.data.Dataset.from_tensor_slices((coords, coords)).batch(batch_size, drop_remainder=True)

##################################################################################

    #####      Architecture      #####
    class FCModel(Model):
      def __init__(self, num_layers, num_ftrs):
        super(FCModel, self).__init__()
        self.dense_layers = []
        for i in range(num_layers):
            self.dense_layers.append(Dense(num_ftrs))
        self.dout = Dense(5)

      def call(self, x):
        x = tf.reshape(x, (batch_size*2, 3))
        for layer in self.dense_layers:
            x = activation(layer(x))
        return self.dout(x)

    # based on "Deep Residual Learning for Nonlinear Regression"
    class ResnetFCIdentity(Layer):
        def __init__(self, nn):
            super(ResnetFCIdentity, self).__init__()
            self.fc1 = Dense(nn)
            self.fc2 = Dense(nn)
            self.fc3 = Dense(nn)

        def call(self, x):
            y = self.fc1(x)
            y = activation(y)

            y = self.fc2(y)
            y = activation(y)

            y = self.fc3(y)
            y = activation(y+x)
            return y

    class ResnetFCDense(Layer):
        def __init__(self, nn):
            super(ResnetFCDense, self).__init__()
            self.fc1 = Dense(nn)
            self.fc2 = Dense(nn)
            self.fc3 = Dense(nn)
            self.fc_side = Dense(nn)

        def call(self, x):
            y = self.fc1(x)
            y = activation(y)

            y = self.fc2(y)
            y = activation(y)

            y = self.fc3(y)

            x = self.fc_side(x)
            y = activation(y+x)
            return y

    class ResnetFC(Model):
        def __init__(self, num_blocks=5, num_ftrs=64):
            super(ResnetFC, self).__init__()
            self.dims = [num_ftrs*2**x for x in range(10)]
            self.blocks = []
            for i in range(num_blocks):
                self.blocks.append(ResnetFCDense(self.dims[i]))
                self.blocks.append(ResnetFCIdentity(self.dims[i]))
                self.blocks.append(ResnetFCIdentity(self.dims[i]))
            self.fc_out = Dense(5)

        def call(self, x):
            x = tf.reshape(x, (batch_size*2, 3))
            for cur_layer in self.blocks:
                x = cur_layer(x)
            x = self.fc_out(x)
            return x


    def calc_order(theta):
        scale = 50
        norm = (theta/np.pi+1)/2

        v1 = (tf.tanh(scale*(norm-0))+1)*0.5
        v2 = (tf.tanh(scale*(norm-1))+1)*0.5
        v3 = (tf.tanh(scale*(norm-2))+1)*0.5
        v4 = (tf.tanh(scale*(norm-3))+1)*0.5
        v = tf.clip_by_value(v1+v2+v3+v4-1, 0, 3)
        coeff = tf.reshape(tf.convert_to_tensor([1., 4., 16., 64., 256.]), (1, 5))
        coeff = tf.repeat(coeff, batch_size, 0)
        v = tf.reduce_sum(v*coeff, 1)
        return v/341.0

    #####      Loss      #####
    def Distance(joints):
        var = -1.5
        d = 0.0

        for o in ObstacleList:
            ocenter = [o.x, o.y, o.z]
            o_r_sqr = o.r**2

            for i in range(0, 6):
                dist_sqr = tf.reduce_sum(tf.square(joints[:,i] - ocenter), axis=1)   #tf.sqrt(
                rev_sigmoid = tf.exp(var*(dist_sqr - o_r_sqr + PENALTY_DISTANCE_EPSILON)) / (1 + tf.exp(var*(dist_sqr - o_r_sqr + PENALTY_DISTANCE_EPSILON)))    
                d = d + rev_sigmoid

            for i in range(1, 6):
                midpoint_joint_pos = (joints[:,i] + joints[:,i-1]) / 2
                dist_sqr = tf.reduce_sum(tf.square(midpoint_joint_pos - ocenter), axis=1)   
                rev_sigmoid = tf.exp(var*(dist_sqr - o_r_sqr + PENALTY_DISTANCE_EPSILON)) / (1 + tf.exp(var*(dist_sqr - o_r_sqr + PENALTY_DISTANCE_EPSILON)))    
                d = d + rev_sigmoid
        #for obstacles end

        return d       


##########################################################################################################

    def FK_loss(gt, pred):
        allJs = fkc.FK(pred)
        obstace_penalty = Distance(allJs)

        pred_joint_pos = allJs[:,-1] #end effector pos
        pred_angle = tf.reshape(pred, (batch_size, 10)) #5+5 angles
        gt_joint_pos = tf.reshape(gt, (batch_size*2, 3)) #*2 because of pairs
        #dist = tf.sqrt(tf.reduce_sum(tf.square(pred_joint_pos-gt_joint_pos), axis=1))
        return obstace_penalty + tf.reduce_mean(tf.losses.mae(gt_joint_pos, pred_joint_pos)) #+0.1*tf.reduce_mean(tf.losses.mse(pred_angle[:,5:], pred_angle[:,:5]))

    #############################################################################
    #
    #############################################################################
    def IsPosBeyondReach(pos, opos, orad):
        sphere = Sphere(opos, orad)
        line = Line([0, 0, 0], pos)

        try:
            a, b = sphere.intersect_line(line)
        except:
            return False    

        if len(a) and len(b):
            if a[1] > b[1]: #looking for top y, b needs to be top point
                tmp = b
                b = a
                a = tmp
    
            sphere_d = orad * np.arccos(np.dot(a-opos,b-opos) / orad**2) #length of arc between points a and b
            # ab_dist = np.sqrt(Distance(a,b))
            # ab_phi = math.asin(ab_dist / 2 / orad)
            # sphere_d = 2 * ab_phi * orad
            total_d = np.sqrt(Distance2([0,0,0], a)) + sphere_d + np.sqrt(Distance2(b, pos))
            if total_d > SUM_OF_ARM_JOINTS_LEN_MM / 1000:
                return True
        
        return False
    #end    

    #############################################################################
    #
    #############################################################################
    def Distance2(a,b):
        return (np.square(b[0]-a[0]) + np.square(b[1]-a[1]) + np.square(b[2]-a[2]))
    #end

    #############################################################################
    #
    #############################################################################
    def IsLinkIntersect(link1, link2, opos, orad):
        d1 = Distance2(link1, opos)
        d2 = Distance2(link2, opos)
        d3 = Distance2((link1+link2)/2, opos) #mid link point distance
        r = orad ** 2
        
        if d1 <= r or d2 <= r or d3 <= r:
            return True
        #endif
        
        return False
    #end

    #############################################################################
    #
    #############################################################################
    def IsArmIntersect(Js, opos, orad):
        for i in range(1,6):
            result = IsLinkIntersect(Js[i], Js[i-1], opos, orad)
            if result == True:
                return True
        #end for

        return False
    #end

    #############################################################################
    #
    #############################################################################
    def ExportDataToCSV(db, filePrefix=""):
        with open(filePrefix+'.csv','w') as out:
            csv_out = csv.writer(out)
            for row in db:
                csv_out.writerow(row)
    #EOFunc    

###############################################################################################################

    #####      Predict on test      #####
    def predict(model_path, test):
        model.load_weights(model_path)

        #save model for later use (ANN to SNN)
        model.build((None, 3))
        model.save_weights(name+".h5", save_format='h5')

        recovered_coord = np.zeros((len(test), batch_size, 2, 3))
        orig_coord = np.zeros_like(recovered_coord)
        recovered_angles = np.zeros((len(test), batch_size*2, 5))
        orig_angles = np.zeros_like(recovered_angles)

        for i,test_batch in enumerate(test.take(len(test))):
            orig_coord[i] = test_batch[0]
            pred_angles = model(orig_coord[i].reshape((batch_size,6)), training=False) #Predict
            recovered_coord[i] = tf.reshape(fkc.FK(pred_angles)[:,-1],(batch_size,2,3))
            recovered_angles[i]=pred_angles
        orig_coord = orig_coord.reshape((-1,3))
        recovered_coord = recovered_coord.reshape((-1,3))
        recovered_angles = recovered_angles.reshape((-1,5))
        return orig_coord, recovered_coord, recovered_angles
    #end


###############################################################################################################
    PREDICT_FUNC = predict
    MODEL_CLASS =  FCModel  #place ResnetFC for ResNet
    MODEL_LOSS_FUNC = FK_loss
    ALIAS_fk_above_1cm_dist = fk_above_1cm_dist 
    ALIAS_fk_above_1mm_dist = fk_above_1mm_dist, 
    ALIAS_fk_mean_dist = fk_mean_dist, 
    ALIAS_fk_std_dist = fk_std_dist, 
    ALIAS_fk_inside_obstacle = fk_inside_obstacle  
    ALIAS_val_fk_above_1cm_dist = 'val_fk_above_1cm_dist'
    ALIAS_val_fk_above_1mm_dist = 'val_fk_above_1mm_dist'        

    #####      Callbacks      #####
    log_dir = "logs/" + name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    val_loss_chkpt = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('checkpoints', name,'val_loss',''),
                                                        monitor='val_loss', save_best_only=True, verbose=False)
    val_1cm_chkpt = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('checkpoints', name, ALIAS_val_fk_above_1cm_dist,''),
                                                       monitor=ALIAS_val_fk_above_1cm_dist, save_best_only=True, verbose=False)
    val_1mm_chkpt = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('checkpoints', name,ALIAS_val_fk_above_1mm_dist,''),
                                                       monitor=ALIAS_val_fk_above_1mm_dist, save_best_only=True, verbose=False)
    lr_callback = RelativeReduceLROnPlateau(monitor='val_fk_mean_dist', factor=0.8, patience=22*f, verbose=0, alpha=0.005,
                                            cooldown=0, min_lr=1e-6)
    early_stopping = EarlyStoppingWMinEpoch(monitor=ALIAS_val_fk_above_1mm_dist, min_delta=0.05, patience=110*f,  earliest_epoch=750)


    #####      Optimizers      #####
    opt = tf.keras.optimizers.SGD(learning_rate=lr_init)

    #####      Compile and fit model      #####
    model = MODEL_CLASS(num_layers, num_ftrs)

    if PREDICT_ONLY == False:
        model.compile(optimizer=opt,
                    loss=MODEL_LOSS_FUNC,
                    metrics=[ALIAS_fk_above_1cm_dist, ALIAS_fk_above_1mm_dist, ALIAS_fk_mean_dist, ALIAS_fk_std_dist, ALIAS_fk_inside_obstacle])
        model.fit(train, validation_data=val, validation_batch_size=100, epochs=num_epochs,verbose=2,
                        callbacks = [tensorboard_callback, lr_callback,
                                    val_loss_chkpt, val_1cm_chkpt, val_1mm_chkpt, early_stopping])


#############################################################################
#
############################################################################# 
 #uncomment for specific model   name = 'fc200kenergy_NO_angles_in_loss2.0cm_obstcl_20210511-115543_5_128_mish_0_10_0.35_0.2_2.5_lrsched2'

    tf.print("\n\n *** Predicting model *** : "+name+"\n")
    best_1cm = PREDICT_FUNC(os.path.join('checkpoints', name, ALIAS_val_fk_above_1cm_dist,''), test)
    best_1mm = PREDICT_FUNC(os.path.join('checkpoints', name, ALIAS_val_fk_above_1mm_dist,''), test)

    insideObstacle = []
    for o in ObstacleList:
        distance_func = lambda t: (np.square(t[0] - o.x)+np.square(t[1] - o.y)+np.square(t[2] - o.z)) <= (o.r ** 2)
        squares = np.array([distance_func(xi) for xi in np_test_x])
        insideObstacle = np.append(insideObstacle, np_test_x[squares])

    insideObstacle = np.reshape(insideObstacle, (-1,3))

    arm   = viper300()
    count_intersections = 0
    colided_ee_points = []
    beyond_reach_ee_points = []
    mm1_test_ee_points = []
    inside_obstacle_ee_points = []
    above_1mm_ee_points = []
    points_beyond_reach = 0
    points_beyond_reach_intersect= 0
    points_inside_obstacle_intersect= 0
    points_beyond_reach_above_1mm = 0
    points_intersect_above_1mm = 0

    for_counter = 0
    for sample in best_1mm[2]: #angles
        sampleq = np.reshape(sample, (5,1))
        Js = arm.calculate_Js(sampleq) #all joints angles

        dist = np.sqrt(np.square(best_1mm[0][for_counter][0]-best_1mm[1][for_counter][0])+np.square(best_1mm[0][for_counter][1]-best_1mm[1][for_counter][1])+np.square(best_1mm[0][for_counter][2]-best_1mm[1][for_counter][2]))
        #np.linalg.norm(best_1cm[0][for_counter]-best_1cm[1][for_counter], 2, axis=1)

        mm1_test_ee_points = np.append(mm1_test_ee_points, best_1mm[0][for_counter]) #original xyz
        mm1_test_ee_points = np.append(mm1_test_ee_points, sample) #original xyz
        if dist > 0.001:
            above_1mm_ee_points = np.append(above_1mm_ee_points, best_1mm[0][for_counter]) #original xyz
            above_1mm_ee_points = np.append(above_1mm_ee_points, sample) #angles

        for o in ObstacleList:
            if (np.square(best_1mm[0][for_counter][0]-o.x)+np.square(best_1mm[0][for_counter][1]-o.y)+np.square(best_1mm[0][for_counter][2]-o.z)) <= (o.r**2):
                inside_obstacle_ee_points = np.append(inside_obstacle_ee_points, best_1mm[0][for_counter]) #original xyz
                inside_obstacle_ee_points = np.append(inside_obstacle_ee_points, sample) #angles            

            ocenter = [o.x, o.y, o.z]
            isbeyond = IsPosBeyondReach(best_1mm[0][for_counter],  ocenter, o.r)
            if isbeyond:
                points_beyond_reach = points_beyond_reach + 1        
                points_beyond_reach_above_1mm = points_beyond_reach_above_1mm + np.sum(dist > 0.001) 
                beyond_reach_ee_points = np.append(beyond_reach_ee_points, best_1mm[0][for_counter]) #original xyz
                beyond_reach_ee_points = np.append(beyond_reach_ee_points, sample) #original xyz
            
            
            isIntersect = IsArmIntersect(Js, ocenter, o.r)
 
            if isIntersect == True: #collision
                count_intersections = count_intersections + 1
                colided_ee_points = np.append(colided_ee_points, best_1mm[0][for_counter]) #original xyz
                colided_ee_points = np.append(colided_ee_points, sample) #predicted colliding angles

                points_intersect_above_1mm = points_intersect_above_1mm + np.sum(dist > 0.001)

                #isbeyond = IsPosBeyondReach(best_1mm[0][for_counter], OBSTACLE_CENTER_POS_IN_M, OBSTACLE_RADIUS_IN_M)
                if isbeyond:
                    points_beyond_reach_intersect = points_beyond_reach_intersect + 1

                if (np.square(best_1mm[0][for_counter][0]-o.x)+np.square(best_1mm[0][for_counter][1]-o.y)+np.square(best_1mm[0][for_counter][2]-o.z)) <= (o.r**2):
                    points_inside_obstacle_intersect = points_inside_obstacle_intersect + 1
        #for o internal 

        for_counter = for_counter + 1
    #end for samples

    colided_ee_points = np.reshape(colided_ee_points, (-1,8)) #xyz + 5 angles
    ExportDataToCSV(list(colided_ee_points),name+'_collision-')
    mm1_test_ee_points = np.reshape(mm1_test_ee_points, (-1,8))
    ExportDataToCSV(list(mm1_test_ee_points),name+'_1mm_test_ee_points-')
    above_1mm_ee_points = np.reshape(above_1mm_ee_points, (-1,8))
    ExportDataToCSV(list(above_1mm_ee_points),name+'_above_1mm_ee_points-')   
    beyond_reach_ee_points = np.reshape(beyond_reach_ee_points, (-1,8))
    ExportDataToCSV(list(beyond_reach_ee_points),name+'_beyond_reach_ee_points-')  
    inside_obstacle_ee_points = np.reshape(inside_obstacle_ee_points, (-1,8))
    ExportDataToCSV(list(inside_obstacle_ee_points),name+'_inside_obstacle_ee_points-') 


    with open(name+'_stats_1cm.txt', 'w') as f:
        dist = np.linalg.norm(best_1cm[0]-best_1cm[1], 2, axis=1)
        f.write('Num of test Samples, Points inside obstacle, Arm intersections cases, Max, Min, Mean, Median, Over 5cm, Over 1cm, Over 1mm, Points Beyond Reach Intersect, Points Beyond Reach\n')
        res_str = '{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(len(np_test_x), len(insideObstacle), count_intersections, np.amax(dist), np.amin(dist), np.mean(dist), np.median(dist),
              np.sum(dist>0.05), np.sum(dist>0.01), np.sum(dist>0.001), points_beyond_reach_intersect, points_beyond_reach)
        f.write(res_str)
        f.write('\nObstacle list:\n')
        ostr = ' '.join(map(str,ObstacleList))
        f.write(ostr)


    with open(name+'_stats_1mm.txt', 'w') as f:
        dist = np.linalg.norm(best_1mm[0]-best_1mm[1], 2, axis=1)
        f.write('Num of test Samples, Points inside obstacle, Arm intersections cases, Max, Min, Mean, Median, Over 5cm, Over 1cm, Over 1mm, Over 1mm excl. inside obstacle, Points Beyond Reach Intersect, Points Beyond Reach, Inside Obstacle Intersect, Intersect inside obstacle + beyond reach, points_beyond_reach_above_1mm, points_intersect_above_1mm \n')
        res_str = '{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(len(np_test_x), len(insideObstacle), count_intersections, np.amax(dist), np.amin(dist), np.mean(dist), np.median(dist),
            np.sum(dist>0.05), np.sum(dist>0.01), np.sum(dist>0.001), np.sum(dist>0.001)-len(insideObstacle), points_beyond_reach_intersect, points_beyond_reach, points_inside_obstacle_intersect, points_inside_obstacle_intersect+points_beyond_reach_intersect,points_beyond_reach_above_1mm, points_intersect_above_1mm )
        f.write(res_str)
        f.write('\nObstacle list:\n')
        ostr = ' '.join(map(str,ObstacleList))
        f.write(ostr)

    print(name)
    print(DATA_FILE)
    print(PENALTY_DISTANCE_EPSILON)
    print(ObstacleList)

#############################################################################
#
#############################################################################
config2 = [
    [5, 128, 4, 0, 10],
]
for c in config2:
    print('Running config: {}'.format(c))
    test_NN_IK_configuration(c[0], c[1], c[2], c[3], c[4])