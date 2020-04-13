import numpy as np
from Bio import SeqIO
import sys
import pandas as pd

def to_binary(seq):
    # eoncode non-standard amino acids like X as all zeros
    # output a array with size of L*20
    seq = seq.upper()
    aas = 'ACDEFGHIKLMNPQRSTVWY'
    pos = dict()
    for i in range(len(aas)): pos[aas[i]] = i
    
    binary_code = dict()
    for aa in aas: 
        code = np.zeros(20)
        code[pos[aa]] = 1
        binary_code[aa] = code
    
    seq_coding = np.zeros((len(seq),20))
    for i,aa in enumerate(seq): 
        code = binary_code.get(aa,np.zeros(20))
        seq_coding[i,:] = code
    return seq_coding

def load_aa_properties(fname):
    df = pd.read_csv(fname,index_col=0)
    pro_code = dict()
    for aa in df.index: pro_code[aa] =  np.around(df.loc[aa,:].values,decimals=3).tolist()
    return pro_code

def to_properties(seq,pro_code):
    # seq: aa sequence
    # pro_code: a dictionary with lists of selected aa properties
    seq_coding = list()
    default = np.zeros(len(pro_code['A'])).tolist()
    for aa in seq: seq_coding.append(pro_code.get(aa,default))
    
    return np.array(seq_coding)
    
    
def zero_padding(inp,length,start=False):
    # zero pad input one hot matrix to desired length
    # start .. boolean if pad start of sequence (True) or end (False)
    assert len(inp) < length
    out = np.zeros((length,inp.shape[1]))
    if start:
        out[-inp.shape[0]:] = inp
    else:
        out[0:inp.shape[0]] = inp
    return out

def zero_padding_center(inp,length):
    # zero pad input one hot matrix to desired length
    # pad equal amount of zeros before and after the sequence.
    assert len(inp) <= length
    out = np.zeros((length,inp.shape[1]))
    
    top_ind = int(np.floor((length-inp.shape[0])/2))
    bot_ind = top_ind + inp.shape[0]
    out[top_ind:bot_ind] = inp 
    
    return out

def random_padding(inp,length):
    # put random zeros before and after sequence
    assert len(inp) <= length
    out = np.zeros((length,inp.shape[1]))
    top_ind = np.random.randint(0,length-inp.shape[0])
    
    bot_ind = top_ind + inp.shape[0]
    out[top_ind:bot_ind] = inp 
    
    return out

def repeat_padding(inp,length):
    # put random zeros before and after sequence
    assert len(inp) <= length
    out = np.zeros((length,inp.shape[1]))
    start = 0
    while start + inp.shape[0] < length:
        out[start:start + inp.shape[0]] = inp 
        start += inp.shape[0]
    out[start:length] = inp[:length-start]
    
    return out

def make_encoding(fname,propfile=None,pad='left',length_cutoff = 2000):
    # fname is the input fasta file
    # pad, either 'left' or 'center'
    # propfile, a csv file with aa properties, if provided then use property encoding, otherwise use one-hot
    
    xseq,y = [],[]
    
    if propfile is not None: pro_code = load_aa_properties(propfile)
    
    for rec in SeqIO.parse(fname,'fasta'):
        #>P43408 ogt=85;topt=70.0
        uni = rec.id
        
        # >Q96XN9 ogt=75;topt=80.0
        if 'topt=' in rec.description: t = float(rec.description.split('=')[-1])
        else: t = float(rec.description.split()[-1])
        seq = rec.seq
        
        if len(seq)>length_cutoff: continue
        if propfile is None: coding = to_binary(seq)
        else: coding = to_properties(seq,pro_code)
        
        ### center or center
        if pad == 'left':   coding = zero_padding(coding,length_cutoff)
        elif pad == 'center': coding = zero_padding_center(coding,length_cutoff)
        else: sys.exit("Pleae check your encoding argument")

        xseq.append(coding)
        
        ###### important to check to swith between topt and ogt 
        y.append(t)
        ######

    xseq = np.array(xseq)
    y = np.array(y).reshape([len(y),1])
    print(xseq.shape,y.shape)
    
    return xseq,y

def split_dataset(X,Y,split=0.9):
    '''create randomly shuffled indices for X and Y
    separate one hot vectors and values'''
    # improve storage efficiency
    np.random.seed(seed=10)
    ind1 = np.random.permutation(np.linspace(0,X.shape[0]-1,X.shape[0],dtype='int32'))
    splt = np.round(X.shape[0]*split)-1
    splt = splt.astype('int64')
    X_train = np.int8(X[ind1[:splt]]) # one hot!!
    X_val = np.int8(X[ind1[splt:]])
    
    
    Y_train = Y[ind1[:splt]]
    Y_val = Y[ind1[splt:]]
    return X_train, X_val, Y_train, Y_val

def load_sequences_ids_labels(fname):
    # fname is a fasta file
    # return 1) a list with all sequence ids; 2) a dictonary with labels and 3) a dictionary with sequences
    list_IDs = []
    labels   = {}
    Seqs     = {}
    for rec in SeqIO.parse(fname,'fasta'):
        list_IDs.append(rec.id)
        if 'topt=' in rec.description: t = float(rec.description.split('=')[-1])
        else: t = float(rec.description.split()[-1])
        labels[rec.id] = t
        Seqs[rec.id]   = str(rec.seq)
    return list_IDs, labels, Seqs

def split_dataset_IDs(list_IDs,split=0.9):
    np.random.seed(seed=10)
    np.random.shuffle(list_IDs)
    splt = np.round(len(list_IDs)*split)-1
    splt = splt.astype('int64')
    
    return list_IDs[:splt], list_IDs[splt:]

def load_train_val(trainfile):
    X, Y = make_encoding(trainfile,pad='left')
    X_train, X_val, Y_train, Y_val = split_dataset(X,Y,split=0.9)
    
    print('Train     :',X_train.shape,Y_train.shape)
    print('Validation:',X_val.shape,  Y_val.shape  )
    return X_train, X_val,  Y_train, Y_val