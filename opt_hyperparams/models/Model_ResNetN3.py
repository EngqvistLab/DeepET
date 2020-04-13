
# coding: utf-8

# In[ ]:


import numpy as np
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Input, Dense, Flatten, Concatenate, Activation, Add
from keras.regularizers import l2
import sys

def Params():
    params = {
        # intial conv
        'filters': [64, 128, 256, 512],    # 128, ok
        'kernel_size1': [3, 7, 9, 11, 21, 31], # 1, ok
        # kernel size of (6 and 10) are close to the average length of beta-sheet and alpha-helix in a protein
        # knowing the full distribution of amino acids that occur at a position (and in its vicinity, 
        # typically ~7 residues on either side) throughout evolution provides a much better picture of the structural 
        # tendencies near that position
        
        # res1
        'kernel_size21':  [3, 7, 9, 11, 21, 31],   # 1, ok
        'kernel_size22':  [3, 7, 9, 11, 21, 31],   # 3, ok
        'dilation2': [1, 2, 3, 5],        # 2, ok
        
        # res2
        'kernel_size31':  [3, 7, 9, 11, 21, 31],   # 1, ok
        'kernel_size32':  [3, 7, 9, 11, 21, 31],   # 3, ok
        'dilation3': [1, 2, 3, 5],                 # 3, ok
        
        # res3
        'kernel_size41':  [3, 7, 9, 11, 21, 31],   # 1, ok
        'kernel_size42':  [3, 7, 9, 11, 21, 31],   # 3, ok
        'dilation4': [1, 2, 3, 5],   
        
        # Pooling
        'pool_size1': [2, 4, 8, 20, 30, 50],  # 3
        
        # dense 1
        'dense1': [128, 256, 512],    # 256 
        
        # dropout 1
        'dropout1': (0,0.5),

        # dense 2
        'dense2': [64, 128, 256, 512],     # 256
        
        # dropout 2
        'dropout2': (0,0.5),
        
        'lr': [1e-4, 5e-4,1e-3],
        'mbatch': [32,64,128,256],
        'patience': [50], 
        'min_delta': [0.01],
        'epochs': [500],
        'res_num': [1,2,3],
        
    }
    
    return {k: np.random.choice(v) if type(v) == list else np.random.uniform(v[0], v[1]) for k,v in params.items()}



def POC_model(input_shape, p):
    input_shape_hot = input_shape[0]
    X_input = Input(shape=input_shape_hot)
    
    def residual_block(data, filters,kernel_size1,kernel_size2, d_rate,index):
        """
          _data: input
          _filters: convolution filters
          _d_rate: dilation rate
        """

        shortcut = data

        bn1 = BatchNormalization(name='res_{0}_bn_1'.format(index))(data)
        act1 = Activation('relu',name='res_{0}_activation_1'.format(index))(bn1)
        conv1 = Conv1D(filters, kernel_size1, dilation_rate=d_rate, padding='same', kernel_regularizer=l2(0.001),
                      name='res_{0}_conv1d_dilated'.format(index))(act1)

        #bottleneck convolution
        bn2 = BatchNormalization(name='res_{0}_bn_2'.format(index))(conv1)
        act2 = Activation('relu',name='res_{0}_activation_2'.format(index))(bn2)
        conv2 = Conv1D(filters, kernel_size2, padding='same', kernel_regularizer=l2(0.001),
                      name='res_{0}_conv1d'.format(index))(act2)

        #skip connection
        x = Add()([conv2, shortcut])

        return x

    #initial conv
    X = Conv1D(int(p['filters']), int(p['kernel_size1']),padding='same',name='conv1d_0')(X_input) 
    
    for i in range(p['res_num']):
        # Residual blocks
        X = residual_block(X, int(p['filters']), int(p['kernel_size{0}1'.format(i+2)]),
                           int(p['kernel_size{0}2'.format(i+2)]),int(p['dilation{0}'.format(i+2)]),i+1)

    X = MaxPooling1D(pool_size=int(p['pool_size1']),padding='same',name='maxpooling1d_1')(X)

    # flatten
    X = Flatten(name='flatten_1')(X)
    
    X = Dense(int(p['dense1']), activation='relu', kernel_initializer='he_uniform',kernel_regularizer=l2(0.0001),name='dense_1')(X)
    X = BatchNormalization()(X)
    X = Dropout(p['dropout1'],name='dropout_1')(X)
    
    X = Dense(int(p['dense2']), activation='relu', kernel_initializer='he_uniform',kernel_regularizer=l2(0.0001),name='dense_2')(X)
    X = BatchNormalization()(X)
    X = Dropout(p['dropout2'],name='dropout_2')(X)
    
    
    # Step 3 - output
    X = Dense(1,name='dense_3')(X)
    model = Model(inputs = X_input, outputs = X)
    
    return model

