
# coding: utf-8

# In[ ]:


import numpy as np
from keras.models import Model
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Input, Dense, Flatten, Concatenate, Activation, Add, AveragePooling1D
from keras.regularizers import l2
import sys


def RES1(input_shape, p):
    input_shape_hot = input_shape[0]
    X_input = Input(shape=input_shape_hot)
    
    def residual_block(data, filters,kernel_size1,kernel_size2, d_rate):
        """
          _data: input
          _filters: convolution filters
          _d_rate: dilation rate
        """

          

        shortcut = data

        bn1 = BatchNormalization()(data)
        act1 = Activation('relu')(bn1)
        conv1 = Conv1D(filters, kernel_size1, dilation_rate=d_rate, padding='same', kernel_regularizer=l2(0.001))(act1)

        #bottleneck convolution
        bn2 = BatchNormalization()(conv1)
        act2 = Activation('relu')(bn2)
        conv2 = Conv1D(filters, kernel_size2, padding='same', kernel_regularizer=l2(0.001))(act2)

        #skip connection
        x = Add()([conv2, shortcut])

        return x

    #initial conv
    conv = Conv1D(int(p['filters']), int(p['kernel_size1']),padding='same')(X_input) 

    # per-residue representation
    
    res = residual_block(conv, int(p['filters']), int(p['kernel_size21']),int(p['kernel_size22']),int(p['dilation2']))
        

    X = MaxPooling1D(pool_size=int(p['pool_size1']))(res)
    X = Dropout(0.5)(X)

    # flatten
    X = Flatten()(X)
    
    X = Dense(int(p['dense1']), activation='relu', kernel_initializer='he_uniform',kernel_regularizer=l2(0.0001))(X)
    X = BatchNormalization()(X)
    X = Dropout(0.5)(X)
    
    X = Dense(int(p['dense2']), activation='relu', kernel_initializer='he_uniform',kernel_regularizer=l2(0.0001))(X)
    X = BatchNormalization()(X)
    X = Dropout(0.5)(X)
    
    
    # Step 3 - output
    X = Dense(1)(X)
    model = Model(inputs = X_input, outputs = X)
    
    return model


def ResNetRed(input_shape, p):
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
    X = BatchNormalization(name='bn_0')(X)
    X = Activation('relu',name='activation_0')(X)
    
    # first dimention reduction with a pooling layer
    if p['pool_type_1'] == 'max':
        X = MaxPooling1D(pool_size=int(p['pool_size1']),padding='same',name='maxpooling1d_1')(X)
    if p['pool_type_1'] == 'avg': 
        X = AveragePooling1D(pool_size=int(p['pool_size1']),padding='same',name='avgpooling1d_1')(X)
    
    
    # residual blocks
    for i in range(p['res_num']):
        # Residual blocks
        X = residual_block(X, int(p['filters']), int(p['kernel_size{0}1'.format(i+2)]),
                           int(p['kernel_size{0}2'.format(i+2)]),int(p['dilation{0}'.format(i+2)]),i+1)
        
        # cov1d for dimension reduction
        if i < p['res_num'] -1:
            X = Conv1D(int(p['filters']), int(p['kernel_size{0}3'.format(i+2)]),strides=(p['stride_{0}3'.format(i+2)],),padding='same',name='conv1d_{0}3'.format(i+2))(X)

    
    
    # second dimention reduction with a pooling layer
    if p['pool_type_2'] == 'max':
        X = MaxPooling1D(pool_size=int(p['pool_size2']),padding='same',name='maxpooling1d_2')(X)
    if p['pool_type_2'] == 'avg': 
        X = AveragePooling1D(pool_size=int(p['pool_size2']),padding='same',name='avgpooling1d_2')(X)
        
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



def ResNetN3(input_shape, p):
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