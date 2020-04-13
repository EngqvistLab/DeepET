#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from keras.optimizers import Adam
from keras.models import Model, load_model
from keras.regularizers import l2
from keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Dropout, Input, Dense, Flatten, Concatenate

import Parameters as param
import Preprocessing as Prep
import my_callbacks


import os
import argparse

def build_bottleneck_model(model, layer_name):
    for layer in model.layers:
        if layer.name == layer_name:
            output = layer.output
    bottleneck_model = Model(model.input, output)
    print('Layers in base model:')
    for layer in model.layers: print(layer.name)
    print('')
    return bottleneck_model

def frozen_layers(model,last_layer_name):
    # 
    print('Before Frozen:')
    for layer in model.layers: print(layer.name,layer.trainable)
    print('')
    for layer in model.layers:
        layer.trainable=False
        if layer.name == last_layer_name: break
    print('After Frozen:')
    for layer in model.layers: print(layer.name,layer.trainable)
        

def main():
    parser = argparse.ArgumentParser(description='Load pre-trained model, directly trained on the new dataset.')
    parser.add_argument('--modelfile', 
                        help='Input pre-trained model model.h5',
                        metavar='')
    parser.add_argument('--trainfile', 
                        help='Input fasta file with sequences for training and validation',
                        metavar='')
    
    parser.add_argument('--testfile',  
                        help='Input fasta file with sequences for test',
                        metavar='')
    
    parser.add_argument('--hyparam', 
                        help='Name of hyperparameter set in Parameters.py', 
                        default=None,
                        metavar='')
    
    parser.add_argument('--layer', 
                        help='The layer before which will be frozen, inlcuded', 
                        default=None,
                        metavar='')
    
    parser.add_argument('--savemodel', 
                        help='If save the best model, default N for False. Other option Y for True', 
                        default='N',
                        metavar='')
    
    parser.add_argument('--tag', 
                        help='tag for log file', 
                        default='',
                        metavar='')
    
    parser.add_argument('--outdir',    
                        help='Directory for output results, including model and logfiles',
                        metavar='')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.outdir): os.mkdir(args.outdir)
    outlog = os.path.join(args.outdir,'train_val_test_history{0}.csv'.format(args.tag))
    
    
    # Load model and parameters
    model = load_model(args.modelfile, custom_objects={'coef_det_k':my_callbacks.coef_det_k})
    
    if args.layer is not None: frozen_layers(model,args.layer)
    
    exec('p = param.{0}()'.format(args.hyparam),None,globals())
        
    # Load data
    X_train, X_val,  Y_train, Y_val = Prep.load_train_val(args.trainfile)
    X_test, Y_test = Prep.make_encoding(args.testfile)
    
    
    # Set callbacks
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, min_lr=0.0000001)
    e_stop = EarlyStopping(monitor='val_loss', min_delta=float(p['min_delta']), patience=200)
    tcb = my_callbacks.TestCallback((X_test,Y_test))
    csvloger = my_callbacks.MyCSVLogger(outlog,tcb)
    
    call_backs = [e_stop, reduce_lr,tcb,csvloger]
    if args.savemodel=='Y':
        checkpoint = ModelCheckpoint(filepath=os.path.join(args.outdir,'bestmodel{0}.h5'.format(args.tag)), save_best_only=True)
        call_backs.append(checkpoint)
    
    # Compile model
    model.compile(optimizer=Adam(lr=p['lr']),loss='mse',metrics=[my_callbacks.coef_det_k])
    print(model.summary())
    
    # Train
    out = model.fit(X_train, Y_train,
                    batch_size=p['mbatch'],
                    epochs=500,
                    validation_data=[X_val,Y_val],
                    callbacks=call_backs)
    
if __name__ == "__main__":
    main()