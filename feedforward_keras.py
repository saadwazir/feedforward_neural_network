import time
import keras
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler,ReduceLROnPlateau
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.datasets import mnist
import tensorflow as tf
gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
session = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

print(gpu_options)
print(session)

class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)




def init_model():
    start_time = time.time()
    print ('Compiling Model ... ')
    model = Sequential()
    model.add(Dense(200, input_dim=361))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(50))
    model.add(Activation('relu'))
              
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    
    model.add(Dense(200))
    model.add(Activation('relu'))
    #model.add(Dropout(0.4))
  
    model.add(Dense(2))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='binary_crossentropy', optimizer=rms, metrics=['accuracy'])
    print ('Model compield in {0} seconds'.format(time.time() - start_time) )
    return model


def run_network(data=None, model=None, epochs=100, batch=32):
    try:
        start_time = time.time()
        
        X_train, X_test, y_train, y_test = data

        if model is None:
            model = init_model()

        history = LossHistory()
        
        def scheduler(epoch):
            if epoch < 10:
                return 0.001
            else:
                return float( 0.001 * tf.math.exp(0.1 * (10 - epoch)) )
            
        def lr_schedule(epoch):
            
            lr = 1e-3
            if epoch > 180:
                lr *= 0.5e-3
            elif epoch > 160:
                lr *= 1e-3
            elif epoch > 120:
                lr *= 1e-2
            elif epoch > 80:
                lr *= 1e-1
            print('Learning rate: ', lr)
            return lr
        
        # checkpoint
        filepath="weights.best.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=2, save_best_only=True, mode='max')
        csv_logger = CSVLogger('training.log')
        scheduler = LearningRateScheduler(scheduler, verbose=1)
        lr_scheduler = LearningRateScheduler(lr_schedule)
        lr_reducer = ReduceLROnPlateau( factor=np.sqrt(0.1),cooldown=0, patience=5, min_lr=0.5e-6)

        print ('Training model...')
        model.fit(X_train, y_train, epochs=epochs, batch_size=batch,
                  callbacks=[history, checkpoint, csv_logger, scheduler  ]
                  , validation_data=[X_test,y_test]
                  , verbose=2)

        print ("Training duration : {0}".format(time.time() - start_time) )
        score = model.evaluate(X_test, y_test, batch_size=16)
        
        
        print('===========================================================')

        print ( "Network's test score [loss, accuracy]: {0}".format(score) )
        return model, history.losses
    except KeyboardInterrupt:
        print (' KeyboardInterrupt')
        return model, history.losses


def plot_losses(losses):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(losses)
    ax.set_title('Loss per batch')
    fig.show()