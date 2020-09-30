#!/usr/bin/env python
# coding: utf-8

# #### 1. try different regression models
# ##### Gang Li, 2018-09-21

# In[2]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression as LR
from sklearn import svm
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import ParameterGrid
import argparse
from Bio import SeqIO

def normalize_dataframe(df):
    # normalize the first n-1 columns. the last colum is the response
    X = df.values[:,:-1]
    #X_n = np.zeros_like(X)
    dfn = df.copy()
    k = 0
    sel_cols = []
    for i in range(X.shape[1]):
        x = X[:,i]
        col = df.columns[i]
        if np.var(x) == 0: continue
        dfn[col] = (x-np.mean(x))/np.var(x)**0.5
    
    return dfn


def do_validation(X,y,model):
    scores = cross_val_score(model,X,y,scoring='r2',cv=5,n_jobs=20)
    res = str(np.mean(scores))+','+str(np.std(scores))+'\n'
    print(res)
    print(scores)
    return res

def split_train_val_test(dfin, test_IDs):
    # dfin, a dataframe with features of all samples inlcuding test samples. features are already normalized
    # test_IDs, a list of sequence IDs of test dataset
    seq_ids = [ind for ind in dfin.index if ind not in test_IDs]
    
    np.random.shuffle(seq_ids)
    
    splt = int(len(seq_ids)*0.9)
    
    train_IDs = seq_ids[:splt]
    val_IDs   = seq_ids[splt:]
    
    dftrain   = dfin.loc[train_IDs,:]
    dfval     = dfin.loc[val_IDs,:]
    dftest    = dfin.loc[test_IDs,:]
    
    return dftrain, dfval, dftest
 
def lr():
    return LR(), {}


def elastic_net():
    param_space = {'l1_ratio':[.1, .5, .7, .9, .95, .99, 1],
                  'alpha': [.1, .2,.4, .6, .8, 1]}
    return ElasticNet(), param_space



def bayesridge():
    param_space = {'alpha_1' :[1e-5, 1e-6, 1e-7],
                   'alpha_2' :[1e-5, 1e-6, 1e-7],
                   'lambda_1':[1e-5, 1e-6, 1e-7],
                   'lambda_2':[1e-5, 1e-6, 1e-7]}
    
    model = BayesianRidge()
    return model, param_space


def svr():
    param_space={
                'C':np.logspace(-5,10,num=16,base=2.0),
                'epsilon':[0,0.01,0.1,0.5,1.0,2.0,4.0]
                }
    model = svm.SVR(kernel='rbf', gamma='auto')
    return model, param_space

def tree():
    param_space={
                'min_samples_leaf':np.linspace(0.01,0.5,10)
                }
    model=DecisionTreeRegressor()
    return model, param_space



def random_forest():
    param_space = {
                    'max_features':np.arange(0.1,1.1,0.1)
    }
    model = RandomForestRegressor(n_estimators=1000,n_jobs=-1)

    return model, param_space

def evaluate(name, model_instance, X_train, Y_train, X_val, Y_val, X_test, Y_test, fhand):
    # param_space = {name:value_lst}
    # model
    lst = []
    best_param = {}
    best_val_score = -np.inf
    test_score_best_model = -np.inf
    
    model, param_space = model_instance()
    for args in list(ParameterGrid(param_space)):
        model.set_params(**args)
        model.fit(X_train, Y_train)
        score = model.score(X_val, Y_val)
        test_score = model.score(X_test, Y_test)
        print(args,'validation r2:',score, 'test r2:',test_score)
        
        if score > best_val_score: 
            best_val_score = score
            best_param     = args
            test_score_best_model = test_score
    
    print('Best model:',best_param, 'validation r2:',best_val_score, 'test r2:',test_score_best_model)
    fhand.write('{0},{1},{2}\n'.format(name, best_val_score, test_score))

def do_clean(df, test_IDs):
    # remove samples with nan feature
    removed_samples = []
    for ind in df.index: 
        if np.isnan(df.loc[ind,:]).any(): removed_samples.append(ind)
    
    sel_inds = [ind for ind in df.index if ind not in removed_samples]
    sel_test_inds = [ind for ind in test_IDs if ind not in removed_samples]
    
    print(len(removed_samples),'were removed due to nan')
    return df.loc[sel_inds,:], sel_test_inds

def test_model_performace(args):
    df = pd.read_csv(args.infile,index_col=0)
    test_IDs = [rec.id for rec in SeqIO.parse(args.testfile,'fasta')]
    df, test_IDs = do_clean(df, test_IDs)
    
    print(df.shape)
    # normalize 
    dfn = normalize_dataframe(df)
    print(dfn.shape)
    
    
    
    dftrain, dfval, dftest = split_train_val_test(dfn, test_IDs)
    
    
    X_train, Y_train = dftrain.values[:,:-1], dftrain.values[:,-1]
    X_val,   Y_val   = dfval.values[:,:-1], dfval.values[:,-1]
    X_test,  Y_test   = dftest.values[:,:-1], dftest.values[:,-1]
    
    print(X_train.shape, X_val.shape, X_test.shape)
    
    
    fhand = open(args.outfile,'w')
    fhand.write('Name,val_r2,test_r2\n')
    evaluate('Linear',       lr,           X_train, Y_train, X_val, Y_val, X_test, Y_test, fhand)
    evaluate('ElasticNet',   elastic_net,  X_train, Y_train, X_val, Y_val, X_test, Y_test, fhand)
    evaluate('BayesRige',    bayesridge,   X_train, Y_train, X_val, Y_val, X_test, Y_test, fhand)
    evaluate('SVR',          svr,          X_train, Y_train, X_val, Y_val, X_test, Y_test, fhand)
    evaluate('Tree',         tree,         X_train, Y_train, X_val, Y_val, X_test, Y_test, fhand)
    evaluate('RandomForest', random_forest,X_train, Y_train, X_val, Y_val, X_test, Y_test, fhand)
    
    fhand.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='''Splite the dataset into train, validation and test datasets. Optimzie the hyperparameters on with validation and test on test dataset''')
    parser.add_argument('--infile',help='a csv file with features of all samples, including test')
    parser.add_argument('--testfile', help='a fasta file with record ids as sample ids in infile')
    parser.add_argument('--outfile', help='a csv file for output results')
    
    args = parser.parse_args()
    test_model_performace(args)
