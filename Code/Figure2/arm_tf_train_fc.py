import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' #'-1' Disable GPU

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import Model
from tensorflow.keras.layers import *
from utils import RelativeReduceLROnPlateau, RelativeEarlyStopping, EarlyStoppingWMinEpoch, FKc
from scipy.spatial import ckdtree
import datetime
import sys

PREDICT_ONLY = False

EPOCHS = 1#80000
PAIR_DISTANCE_IN_M = 0.02

DATA_FILE = "end_effector_test-200000.csv"

def test_NN_IK_configuration(num_layers, num_ftrs, activation, use_angle_norm, batch_size):

    num_layers = num_layers
    num_ftrs = num_ftrs
    activation_name = ['lerelu', 'relu', 'tanh', 'swish', 'mish'][activation]
    activation = [tf.nn.leaky_relu, tf.nn.relu, tf.nn.tanh,  tf.keras.activations.swish, tfa.activations.mish][activation]
    angle_norm=float(use_angle_norm)*1.e-5
    norm_type = use_angle_norm
    batch_size = batch_size
    f=1

    name = 'fc200k_energy_mae_{num_layers}_{num_ftrs}_{activ}_{theta_norm}_{bs}_lrsched2'.\
        format(num_layers=num_layers,num_ftrs=num_ftrs, activ=activation_name, 
               theta_norm=norm_type, bs=batch_size)

    lr_init = 1e-1
    num_epochs = EPOCHS
    fkc = FKc(batch_size*2)

    #################################################################################
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

       
    #################################################################################
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
    def FK_loss(gt, pred):

        pred_joint_pos = fkc.FK(pred)[:,-1,:]
        pred_angle = tf.reshape(pred, (batch_size, 10)) #5+5 angles
        gt_joint_pos = tf.reshape(gt, (batch_size*2, 3)) #*2 because of pairs
        #dist = tf.sqrt(tf.reduce_sum(tf.square(pred_joint_pos-gt_joint_pos), axis=1))
        return tf.reduce_mean(tf.losses.mae(gt_joint_pos, pred_joint_pos))+\
                0.1*tf.reduce_mean(tf.losses.mse(pred_angle[:,5:], pred_angle[:,:5]))

###############################################################################################################

    #####      Callbacks      #####
    log_dir = "logs/" + name + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    val_loss_chkpt = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('checkpoints', name,'val_loss',''),
                                                        monitor='val_loss', save_best_only=True, verbose=False)
    val_1cm_chkpt = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('checkpoints', name,'val_fk_above_1cm_dist',''),
                                                       monitor='val_fk_above_1cm_dist', save_best_only=True, verbose=False)
    val_1mm_chkpt = tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('checkpoints', name,'val_fk_above_1mm_dist',''),
                                                       monitor='val_fk_above_1mm_dist', save_best_only=True, verbose=False)
    lr_callback = RelativeReduceLROnPlateau(monitor='val_fk_mean_dist', factor=0.8, patience=22*f, verbose=0, alpha=0.005,
                                            cooldown=0, min_lr=1e-6)
    early_stopping = EarlyStoppingWMinEpoch(monitor='val_fk_above_1mm_dist', min_delta=0.05, patience=110*f,  earliest_epoch=750)


    #####      Optimizers      #####
    opt = tf.keras.optimizers.SGD(learning_rate=lr_init)


    #####      Compile and fit model      #####
    model = FCModel(num_layers, num_ftrs)

    if PREDICT_ONLY == False:
        model.compile(optimizer=opt,
                    loss=FK_loss,
                    metrics=[fk_above_1cm_dist, fk_above_1mm_dist, fk_mean_dist, fk_std_dist])
        model.fit(train, validation_data=val, validation_batch_size=100, epochs=num_epochs,verbose=2,
                        callbacks = [tensorboard_callback, lr_callback,
                                    val_loss_chkpt, val_1cm_chkpt, val_1mm_chkpt, early_stopping])


###############################################################################################################

    #####      Predict on test      #####
    def predict(model_path, test):
        model.load_weights(model_path)

        #Save model for later ANN-> SNN Nengo conversion
        model.build((None, 3))
        model.save_weights(name+".h5", save_format='h5')

        recovered_coord = np.zeros((len(test), batch_size, 2, 3))
        orig_coord = np.zeros_like(recovered_coord)
        recovered_angles = np.zeros((len(test), batch_size*2, 5))
        orig_angles = np.zeros_like(recovered_angles)
        # Should be a better way!
        for i,test_batch in enumerate(test.take(len(test))):
            orig_coord[i] = test_batch[0]
            pred_angles = model(orig_coord[i].reshape((batch_size,6)), training=False)
            recovered_coord[i] = tf.reshape(fkc.FK(pred_angles)[:,-1],(batch_size,2,3))
#            orig_angles[i] = test_batch[1]
            recovered_angles[i]=pred_angles
        orig_coord = orig_coord.reshape((-1,3))
        recovered_coord = recovered_coord.reshape((-1,3))
#        orig_angles = orig_angles.reshape((-1,5))
        recovered_angles = recovered_angles.reshape((-1,5))
        return orig_coord, recovered_coord, recovered_angles
    #end


#############################################################################
#
############################################################################# 
    tf.print("\n\n *** Predicting model *** : "+name+"\n")
    best_1cm = predict(os.path.join('checkpoints', name,'val_fk_above_1cm_dist',''), test)
    best_1mm = predict(os.path.join('checkpoints', name,'val_fk_above_1mm_dist',''), test)

    with open(name+'_stats_1cm.txt', 'w') as f:
        dist = np.linalg.norm(best_1cm[0]-best_1cm[1], 2, axis=1)
        f.write('Max, Min, Mean, Median, Over 5cm, Over 1cm, Over 1mm\n')
        res_str = '{},{},{},{},{},{},{}\n'.format(np.amax(dist), np.amin(dist), np.mean(dist), np.median(dist),
              np.sum(dist>0.05), np.sum(dist>0.01), np.sum(dist>0.001))
        f.write(res_str)

    with open(name+'_stats_1mm.txt', 'w') as f:
        dist = np.linalg.norm(best_1mm[0]-best_1mm[1], 2, axis=1)
        f.write('Max, Min, Mean, Median, Over 5cm, Over 1cm, Over 1mm\n')
        res_str = '{},{},{},{},{},{},{}\n'.format(np.amax(dist), np.amin(dist), np.mean(dist), np.median(dist),
              np.sum(dist>0.05), np.sum(dist>0.01), np.sum(dist>0.001))
        f.write(res_str)


#############################################################################
#
#############################################################################
config2 = [
    [5, 128, 0, 0, 10], #Init parameters for (num_layers, num_ftrs, activation, use_angle_norm, batch_size)
]
for c in config2:
    print('Running config: {}'.format(c))
    test_NN_IK_configuration(c[0], c[1], c[2], c[3], c[4])