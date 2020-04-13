#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from keras.optimizers import Adam
from keras import backend as K

import Preprocessing as Prep
import my_callbacks

import os
import argparse


def load_train_val(trainfile, propfile,split=0.9):
    X, Y = Prep.make_encoding(trainfile,pad='left',propfile=propfile)
    X_train, X_val, Y_train, Y_val = Prep.split_dataset(X,Y,split=split)
    
    print('Train     :',X_train.shape,Y_train.shape)
    print('Validation:',X_val.shape,  Y_val.shape  )
    return X_train, X_val,  Y_train, Y_val

def load_train_val_test(trainfile, propfile):
    X, Y = Prep.make_encoding(trainfile,pad='left',propfile=propfile)
    X_train, X_test, Y_train, Y_test = Prep.split_dataset(X,Y,split=0.9)
    X_train, X_val, Y_train, Y_val= Prep.split_dataset(X_train,Y_train,split=0.9)
    
    print('Train     :',X_train.shape,Y_train.shape)
    print('Validation:',X_val.shape,  Y_val.shape  )
    print('Test      :',X_test.shape,  Y_test.shape )
    return X_train, X_val, X_test,  Y_train, Y_val, Y_test

    
def train(args,nn,tested_params,X_train, X_val,  Y_train, Y_val):
    # random get p
    p = nn.Params()
    p['patience'] = args.patience
    if p['dense1']<p['dense2']: p['dense1'] = p['dense2']
    
    pset = list(p.items())
    pset.sort()   
    pset = tuple(pset)
    
    if not tested_params.get(pset,False):

        # 1.3 Set callbacks
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, min_lr=0.0000001)
        e_stop = EarlyStopping(monitor='val_loss', min_delta=float(p['min_delta']), patience=int(p['patience']))
        #checkpoint = ModelCheckpoint(filepath=os.path.join(args.outdir,'bestmodel.h5'), save_best_only=True)
        #tcb = my_callbacks.TestCallback((X_test,Y_test))
        csvloger = my_callbacks.MyCSVLogger(filename=outlog,hpars=p,append=True)
        call_backs = [e_stop, reduce_lr,csvloger]

        # 1.4 compile model


        model = nn.POC_model((X_train.shape[1:],),p)
        model.compile(optimizer=Adam(lr=p['lr']), loss='mse',metrics=[my_callbacks.coef_det_k])

        print(model.summary())

        out = model.fit(X_train, Y_train,
                        batch_size=p['mbatch'],
                        epochs=p['epochs'],
                        validation_data=[X_val,Y_val],
                        callbacks=call_backs)
        
        tested_params[pset] = True
        K.clear_session()
        del model
        del out
    return tested_params
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PreTrain on OGT dataset from scratch')
    parser.add_argument('--trainfile', 
                        help='Input fasta file with sequences for training and validation',
                        metavar='')
    
    parser.add_argument('--modelname', 
                        help='Model name in ./models',
                        metavar='')
    
    parser.add_argument('--propfile', 
                        help='A csv file that contain a list of standardized properties for each amino acid, default None. If provided, then will use property encoding instead of onehot encoding',
                        default=None,
                        metavar='')
    
    parser.add_argument('--iterations',    
                        help='Number of random sampling',
                        type=int,
                        metavar='')
    
    parser.add_argument('--patience',    
                        help='patience',
                        type=int,
                        default=50,
                        metavar='')
    
    parser.add_argument('--outdir',    
                        help='Directory for output results, including model and logfiles',
                        metavar='')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.outdir): os.mkdir(args.outdir)
    outlog = os.path.join(args.outdir,'train_val_test_history.csv')
    
    # load model
    exec('from models import {0} as nn'.format(args.modelname.replace('.py','')),None,globals())
        
    
    # 1.2 load train, val, test datasets
    X_train, X_val, Y_train, Y_val = load_train_val(args.trainfile,args.propfile,split=0.8)
    
    tested_params = dict()
    for i in range(args.iterations): 
        # try except seems to be not able to handle OOM error
        tested_params= train(args,nn,tested_params,X_train, X_val, Y_train, Y_val)
                  

