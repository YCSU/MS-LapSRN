from msLapSRN_model import net
from dataset import ReadSequence

import pickle
import tensorflow as tf
import numpy as np
import keras.backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler


def L1_Charbonnier_loss(y_true, y_pred):
    """L1 Charbonnierloss."""
    eps = 1e-6
    y_true = tf.convert_to_tensor(y_true, np.float32)
    y_pred = tf.convert_to_tensor(y_pred, np.float32)
    diff = y_true-y_pred
    error = K.sqrt( diff * diff + eps )
    loss = K.sum(error) 
    return loss

def PSNR(y_true, y_pred):
    y_true = tf.convert_to_tensor(y_true, np.float32)
    y_pred = tf.convert_to_tensor(y_pred, np.float32)
    diff = y_true-y_pred
    rmse = K.sqrt(K.mean(diff * diff))
    return 20.0 * K.log(255 / rmse) / K.log(10.0)


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = 0.000001 * (0.1 ** (epoch // 10))
    return lr

if __name__ == "__main__":

    # load shared weights model, the parameters setting is in the 
    model = net()
   

    # training
    model.compile(optimizer='adam',
                  loss=L1_Charbonnier_loss,
                  metrics=[PSNR])
    lrate=LearningRateScheduler(adjust_learning_rate)

    filepath="./checkpoint/{epoch:02d}-{loss:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, save_weights_only=True, mode='min', period=1)
    callbacks_list = [checkpoint, lrate]

    data = ReadSequence("./data.h5", batch_size=4)
    history = model.fit_generator(data, epochs=100, verbose=1, callbacks=callbacks_list)

    with open('checkpoint/trainHistory.pkl', 'wb') as f:
        pickle.dump(history.history, f)