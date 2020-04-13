#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
from keras.callbacks import ModelCheckpoint, EarlyStopping,ReduceLROnPlateau
from keras.optimizers import Adam

import Preprocessing as Prep
import my_callbacks
import Save

import os
import argparse



def main():
    parser = argparse.ArgumentParser(description='PreTrain on OGT dataset from scratch')
    parser.add_argument('--trainfile', 
                        help='Input fasta file with sequences for training',
                        metavar='')
    
    parser.add_argument('--valfile', 
                        help='Input fasta file with sequences for validation',
                        default=None,
                        metavar='')
    
    
    parser.add_argument('--modelname', 
                        help='Model name in models.py', 
                        default=None, 
                        metavar='')
    
    parser.add_argument('--hyparam', 
                        help='Name of hyperparameter set in Parameters.py', 
                        default=None,
                        metavar='')
    
    parser.add_argument('--mbatch', 
                        help='batch size', 
                        default=0,
                        type=int,
                        metavar='')
    
    parser.add_argument('--lr', 
                        help='learning rate', 
                        default=0,
                        type=float,
                        metavar='')
    
    parser.add_argument('--dense1', 
                        help='size of the first dense layer', 
                        default=0,
                        type=float,
                        metavar='')
    
    parser.add_argument('--generator', 
                        help='Generator name in Generators.py. Default None', 
                        default=None, 
                        metavar='')
    
    
    parser.add_argument('--patience', 
                        help='patience. Default None', 
                        type=int,
                        default=0, 
                        metavar='')
    
    parser.add_argument('--outdir',    
                        help='Directory for output results, including best model file and logfile',
                        metavar='')
    
    args = parser.parse_args()
    print(args)
    
    if not os.path.exists(args.outdir): os.mkdir(args.outdir)
    outlog = os.path.join(args.outdir,'train_val_test_history.csv')
    
    # 1. load data
    # 1.1 load parameters
    exec('from Parameters import {0} as param'.format(args.hyparam),None,globals())
    p = param()
    
    if args.mbatch >0: p['mbatch'] = args.mbatch
    if args.lr > 0: p['lr'] = args.lr
    if args.dense1 > 0: p['dense1'] = args.dense1
    if args.patience > 0: p['patience'] = args.patience
    
    # 1.2 load train, val, test datasets
    
    if args.generator is not None:
        exec('from Generators import {0} as generator'.format(args.generator),None,globals())
        IDs_train, labels_train, Seqs_train = Prep.load_sequences_ids_labels(args.trainfile)
        IDs_val,   labels_val,   Seqs_val   = Prep.load_sequences_ids_labels(args.valfile)

        # 1.4 build generators

        train_generator = generator(IDs_train, 
                                    labels_train, 
                                    Seqs_train, 
                                    p['mbatch'])

        val_generator   = generator(IDs_val, 
                                    labels_val, 
                                    Seqs_val, 
                                    p['mbatch'])
    else: X_train, X_val,  Y_train, Y_val = Prep.load_train_val(args.trainfile)
    
    
    # 1.5 callbacks
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.2, patience=10, min_lr=0.0000001)
    e_stop = EarlyStopping(monitor='val_loss', min_delta=float(p['min_delta']), patience=int(p['patience']))
    best_checkpoint = ModelCheckpoint(filepath=os.path.join(args.outdir,'bestmodel.h5'), save_best_only=True)
    #last_checkpoint = ModelCheckpoint(filepath=os.path.join(args.outdir,'lastmodel.h5'), save_best_only=False)
    #tcb = my_callbacks.TestCallback((X_test,Y_test))
    csvloger = my_callbacks.MyCSVLogger(outlog)
    call_backs = [e_stop, reduce_lr,csvloger,best_checkpoint]
    
    
    # 1.6 compile model
    exec('from models import {0} as nn'.format(args.modelname),None,globals())
    model = nn(((2000,20),),p)
    model.compile(optimizer=Adam(lr=p['lr']), loss='mse',metrics=[my_callbacks.coef_det_k])
    print(model.summary())
    
    
    # 1.7 train
    if args.generator is not None:
        out = model.fit_generator(generator=train_generator,
                                  epochs=p['epochs'],
                                  validation_data=val_generator,
                                  callbacks=call_backs,
                                  use_multiprocessing=True,
                                  workers=0)
    else:
        out = model.fit(X_train, Y_train,
                    batch_size=p['mbatch'],
                    epochs=500,
                    validation_data=[X_val,Y_val],
                    callbacks=call_backs)
    

if __name__ == "__main__":
    main()
