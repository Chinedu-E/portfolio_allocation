from keras import backend as K 
from keras.layers import LeakyReLU, Dense, Conv1D, Dropout, MaxPooling2D, Conv2DTranspose, MaxPooling1D, Input, TimeDistributed, Conv2D, Conv1DTranspose
from keras.models import Model
from keras.optimizers import Adam
import numpy as np


class AE:
    
    def __init__(self) -> None:
        self.z_size=30
        self.model = self.build_model()
        
    def _build_model(self):
        #building encoder
        inputs = Input(shape=(60, 16, 4))
        x = Conv2D(64, kernel_size=3, strides=2, padding="valid")(inputs)
        x = Dropout(0.2)(x)
        x = LeakyReLU(0.2)(x)
        x = MaxPooling2D()(x)
        z = TimeDistributed(Dense(self.z_size))(x)
        # Building decoder
        x = Conv2DTranspose(4, kernel_size=8, strides=4)(z)
        x = LeakyReLU(0.2)(x)
        model = Model(inputs, x)
        return model
    
    def build_model(self):
        #building encoder
        inputs = Input(shape=(60, 64))
        x = Conv1D(64, kernel_size=3, strides=2, padding="valid")(inputs)
        x = Dropout(0.2)(x)
        x = LeakyReLU(0.2)(x)
        x = MaxPooling1D()(x)
        z = TimeDistributed(Dense(self.z_size))(x)
        # Building decoder
        x = Conv1DTranspose(64, kernel_size=8, strides=4)(z)
        x = LeakyReLU(0.2)(x)
        model = Model(inputs, x)
        return model
    
    def compile(self, learning_rate):
        def ae_loss(y_true, y_pred):
            loss = K.mean(K.square(y_true-y_pred), axis = [1, 2])
            return loss
            
        optimizer = Adam(learning_rate)
        self.model.compile(optimizer=optimizer, loss=ae_loss)

def main():
    ...
    
    
    
if __name__=="__main__":
    model = AE().model
    print(model.summary())
 
 