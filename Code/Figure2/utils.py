import tensorflow as tf
import numpy as np
from tensorflow.keras.callbacks import Callback
from tensorflow.python.keras import backend as K

# adapted from ReduceLROnPlateau
class RelativeReduceLROnPlateau(Callback):
    def __init__(self,
                 monitor='val_loss',
                 factor=0.1,
                 patience=10,
                 verbose=0,
                 alpha=0.01,
                 cooldown=0,
                 min_lr=0,
                 **kwargs):
        super(RelativeReduceLROnPlateau, self).__init__()

        self.monitor = monitor
        if factor >= 1.0:
            raise ValueError('RelativeReduceLROnPlateau ' 'does not support a factor >= 1.0.')
        self.factor = factor
        self.min_lr = min_lr
        self.alpha = alpha
        self.patience = patience
        self.verbose = verbose
        self.cooldown = cooldown
        self.cooldown_counter = 0  # Cooldown counter.
        self.wait = 0
        self.best = 0
        self.monitor_op = None
        self.mode = 'min'
        self._reset()

    def _reset(self):
        """Resets wait counter and cooldown counter.
        """
        if (self.mode == 'min' or
                (self.mode == 'auto' and 'acc' not in self.monitor)):
            self.monitor_op = lambda a, b: np.less(a, b*(1.-self.alpha))
            self.best = np.Inf
        else:
            raise Exception('Not implemented')
        self.cooldown_counter = 0
        self.wait = 0

    def on_train_begin(self, logs=None):
        self._reset()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            pass
        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = float(K.get_value(self.model.optimizer.lr))
                    if old_lr > self.min_lr:
                        new_lr = old_lr*self.factor
                        new_lr = max(new_lr, self.min_lr)
                        K.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: RelativeReduceLROnPlateau reducing learning '
                                  'rate to %s.'%(epoch+1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0

    def in_cooldown(self):
        return self.cooldown_counter > 0

# adapted from EarlyStopping
class RelativeEarlyStopping(Callback):
  def __init__(self,
               monitor='val_loss',
               alpha=0.0,
               patience=0,
               earliest_epoch=0,
               verbose=0):
    super(RelativeEarlyStopping, self).__init__()
    self.monitor = monitor
    self.patience = patience
    self.verbose = verbose
    self.alpha = abs(alpha)
    self.wait = 0
    self.stopped_epoch = 0
    self.earliest_epcoh=earliest_epoch
    self.monitor_op = lambda a, b: np.less(a, b*(1.-self.alpha))


  def on_train_begin(self, logs=None):
    # Allow instances to be re-used
    self.wait = 0
    self.stopped_epoch = 0
    self.best = np.Inf

  def on_epoch_end(self, epoch, logs=None):
    current = self.get_monitor_value(logs)
    if current is None:
      return
    if self.monitor_op(current, self.best):
      self.best = current
      self.wait = 0
    else:
      self.wait += 1
      if epoch>=self.earliest_epcoh and self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0 and self.verbose > 0:
      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

  def get_monitor_value(self, logs):
    logs = logs or {}
    monitor_value = logs.get(self.monitor)
    return monitor_value

class EarlyStoppingWMinEpoch(Callback):
  def __init__(self,
               monitor='val_loss',
               min_delta=0,
               patience=0,
               verbose=0,
               mode='auto',
               baseline=None,
               restore_best_weights=False,
               earliest_epoch=0):
    super(EarlyStoppingWMinEpoch, self).__init__()

    self.monitor = monitor
    self.patience = patience
    self.verbose = verbose
    self.baseline = baseline
    self.min_delta = abs(min_delta)
    self.wait = 0
    self.stopped_epoch = 0
    self.restore_best_weights = restore_best_weights
    self.best_weights = None
    self.earliest_epoch = earliest_epoch

    if mode not in ['auto', 'min', 'max']:
      mode = 'auto'

    if mode == 'min':
      self.monitor_op = np.less
    elif mode == 'max':
      self.monitor_op = np.greater
    else:
      if 'acc' in self.monitor:
        self.monitor_op = np.greater
      else:
        self.monitor_op = np.less

    if self.monitor_op == np.greater:
      self.min_delta *= 1
    else:
      self.min_delta *= -1

  def on_train_begin(self, logs=None):
    # Allow instances to be re-used
    self.wait = 0
    self.stopped_epoch = 0
    if self.baseline is not None:
      self.best = self.baseline
    else:
      self.best = np.Inf if self.monitor_op == np.less else -np.Inf
    self.best_weights = None

  def on_epoch_end(self, epoch, logs=None):
    current = self.get_monitor_value(logs)
    if current is None:
      return
    if self.monitor_op(current - self.min_delta, self.best):
      self.best = current
      self.wait = 0
      if self.restore_best_weights:
        self.best_weights = self.model.get_weights()
    else:
      self.wait += 1
      if epoch>=self.earliest_epoch and self.wait >= self.patience:
        self.stopped_epoch = epoch
        self.model.stop_training = True
        if self.restore_best_weights:
          if self.verbose > 0:
            print('Restoring model weights from the end of the best epoch.')
          self.model.set_weights(self.best_weights)

  def on_train_end(self, logs=None):
    if self.stopped_epoch > 0 and self.verbose > 0:
      print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

  def get_monitor_value(self, logs):
    logs = logs or {}
    monitor_value = logs.get(self.monitor)
    return monitor_value

class FKc:
    def __init__(self, bs):
        self.batch_size = bs

        self.l1 = (127-9)*1e-3
        self.l2 = (427-127)*1e-3
        self.l3 = (60)*1e-3
        self.l4 = (253-60)*1e-3
        self.l5 = (359-253)*1e-3
        self.l6 = (567-359)*1e-3

        self.identity_flattened = tf.repeat(tf.reshape(tf.eye(4), (1, 16)), self.batch_size, 0)
        # Note: it would be easier to store everything in a single tensor with dimensions Bs x 6,
        # but we want to preserve naming
        self.l1_batch = tf.repeat([self.l1], self.batch_size,0)
        self.l2_batch = tf.repeat([self.l2], self.batch_size,0)
        self.l3_batch = tf.repeat([self.l3], self.batch_size,0)
        self.l4_batch = tf.repeat([self.l4], self.batch_size,0)
        self.l5_batch = tf.repeat([self.l5], self.batch_size,0)
        self.l6_batch = tf.repeat([self.l6], self.batch_size,0)

    # return every joint position after applying forward kinematics
    '''
    Flattened transforms
    T1 = [sp.cos(q0), 0, sp.sin(q0), 0, 0, 1, 0, 0, -sp.sin(q0), 0, sp.cos(q0), 0, 0, 0, 0, 1]
    T2 = [sp.cos(q1), -sp.sin(q1), 0, 0, sp.sin(q1), sp.cos(q1), 0, l1, 0, 0, 1, 0, 0, 0, 0, 1]
    T3 = [sp.cos(q2), -sp.sin(q2), 0, l3, sp.sin(q2), sp.cos(q2), 0, l2, 0, 0, 1, 0, 0, 0, 0, 1]
    T4 = [1, 0, 0, l4, 0, sp.cos(q3), -sp.sin(q3), 0, 0, sp.sin(q3), sp.cos(q3), 0, 0, 0, 0, 1]
    T45 = [sp.cos(q4), -sp.sin(q4), 0, l5, sp.sin(q4), sp.cos(q4), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    T56 = [1, 0, 0, l6, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
    '''
    def FK(self, angles):
        pos = tf.reshape(tf.constant([0, 0, 0, 1], dtype=float), (4, 1))
        s = tf.sin(angles)
        c = tf.cos(angles)

        # T1 = [sp.cos(q0), 0, sp.sin(q0), 0, 0, 1, 0, 0, -sp.sin(q0), 0, sp.cos(q0), 0, 0, 0, 0, 1]
        l = [c[:, 0], self.identity_flattened[:, 1], s[:, 0]]+tf.unstack(self.identity_flattened[:, 3:8], axis=1)+[-s[:, 0], self.identity_flattened[:, 9], c[:, 0]]+tf.unstack(self.identity_flattened[:, 11:], axis=1)
        T1 = tf.reshape(tf.stack(l, axis=1), (self.batch_size, 4, 4))

        # T2 = [sp.cos(q1), -sp.sin(q1), 0, 0, sp.sin(q1), sp.cos(q1), 0, l1, 0, 0, 1, 0, 0, 0, 0, 1]
        l = [c[:, 1], -s[:,1]] + tf.unstack(self.identity_flattened[:, 2:4], axis=1)+[s[:, 1], c[:,1]]+[self.identity_flattened[:, 6]]+\
             [self.l1_batch]+tf.unstack(self.identity_flattened[:, 8:], axis=1)
        T2 = tf.reshape(tf.stack(l, axis=1), (self.batch_size, 4, 4))
        T2 = tf.matmul(T1, T2)

        # T3 = [sp.cos(q2), -sp.sin(q2), 0, l3, sp.sin(q2), sp.cos(q2), 0, l2, 0, 0, 1, 0, 0, 0, 0, 1]
        l = [c[:, 2], -s[:,2]] + [self.identity_flattened[:, 2]] + [self.l3_batch] +[s[:, 2], c[:,2]]+[self.identity_flattened[:, 6]]+\
             [self.l2_batch]+tf.unstack(self.identity_flattened[:, 8:], axis=1)
        T3 = tf.reshape(tf.stack(l, axis=1), (self.batch_size, 4, 4))
        T3 = tf.matmul(T2, T3)

        # T4 = [1, 0, 0, l4, 0, sp.cos(q3), -sp.sin(q3), 0, 0, sp.sin(q3), sp.cos(q3), 0, 0, 0, 0, 1]
        l = tf.unstack(self.identity_flattened[:, :3], axis=1) + [self.l4_batch] + [self.identity_flattened[:, 4]]+\
            [c[:,3], -s[:,3]] + tf.unstack(self.identity_flattened[:, 7:9], axis=1)+[s[:, 3], c[:,3]]+tf.unstack(self.identity_flattened[:, 11:], axis=1)
        T4 = tf.reshape(tf.stack(l, axis=1), (self.batch_size, 4, 4))
        T4 = tf.matmul(T3, T4)

        #T45 = [sp.cos(q4), -sp.sin(q4), 0, l5, sp.sin(q4), sp.cos(q4), 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        l = [c[:, 4], -s[:,4]] + [self.identity_flattened[:, 2]] + [self.l5_batch] +[s[:, 4], c[:,4]]+tf.unstack(self.identity_flattened[:, 6:], axis=1)
        T5 = tf.reshape(tf.stack(l, axis=1), (self.batch_size, 4, 4))
        T5 = tf.matmul(T4, T5)

        #T56 = [1, 0, 0, l6, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
        l = tf.unstack(self.identity_flattened[:, :3], axis=1)+ [self.l6_batch] + tf.unstack(self.identity_flattened[:, 4:], axis=1)
        T6 = tf.reshape(tf.stack(l, axis=1), (self.batch_size, 4, 4))
        T6 = tf.matmul(T5, T6)

        J1 = tf.matmul(T1, pos)
        J2 = tf.matmul(T2, pos)
        J3 = tf.matmul(T3, pos)
        J4 = tf.matmul(T4, pos)
        J5 = tf.matmul(T5, pos)
        J6 = tf.matmul(T6, pos)

        packed_joint_pos = tf.reshape(tf.stack([J1, J2, J3, J4, J5, J6], axis=1), (self.batch_size, 6,4))[:,:,:3]

        return packed_joint_pos

