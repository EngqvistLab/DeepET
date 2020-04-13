import numpy as np
import keras
import Preprocessing as pre

class DataGenerator(keras.utils.Sequence):
    # Adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    def __init__(self, list_IDs, labels, Seqs, batch_size, weights=None, shuffle=True,length_cutoff=2000):
        self.list_IDs      = list_IDs
        self.labels        = labels     # a dictonary
        self.batch_size    = batch_size
        self.shuffle       = shuffle
        self.Seqs          = Seqs
        self.length_cutoff = length_cutoff
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
           
    
    def __data_generation(self,list_IDs_temp):
        xseq, y = [], []
        for seqid in list_IDs_temp:
            seq = self.Seqs[seqid]
            coding = pre.to_binary(seq)
            coding = pre.zero_padding(coding,self.length_cutoff)
            xseq.append(coding)
            y.append(self.labels[seqid])
        xseq = np.array(xseq)
        y = np.array(y).reshape([len(y),1])
        return xseq, y


class RandomPadding(keras.utils.Sequence):
    # Adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    def __init__(self, list_IDs, labels, Seqs, batch_size, fold, shuffle=True,length_cutoff=2000):
        self.list_IDs      = list_IDs
        self.labels        = labels     # a dictonary
        self.batch_size    = batch_size
        self.shuffle       = shuffle
        self.Seqs          = Seqs
        self.length_cutoff = length_cutoff
        self.fold          = fold # number of batches per epoch given by fold*len(self.list_IDs) / self.batch_size
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size)*self.fold)
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch, random generate batch_size samples
        #indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        indexes = np.random.choice(np.arange(len(self.list_IDs)),size=self.batch_size, replace=False)
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
           
    def __data_generation(self,list_IDs_temp):
        xseq, y = [], []
        for seqid in list_IDs_temp:
            seq = self.Seqs[seqid]
            coding = pre.to_binary(seq)
            # put random number of zeros before and after sequence encoding
            coding = pre.random_padding(coding,self.length_cutoff)
            xseq.append(coding)
            y.append(self.labels[seqid])
        xseq = np.array(xseq)
        y = np.array(y).reshape([len(y),1])
        return xseq, y
    
class SeqRepeat(keras.utils.Sequence):
    # Adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    def __init__(self, list_IDs, labels, Seqs, batch_size, shuffle=True,length_cutoff=2000):
        self.list_IDs      = list_IDs
        self.labels        = labels     # a dictonary
        self.batch_size    = batch_size
        self.shuffle       = shuffle
        self.Seqs          = Seqs
        self.length_cutoff = length_cutoff
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch, random generate batch_size samples
        #indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        indexes = np.random.choice(np.arange(len(self.list_IDs)),size=self.batch_size, replace=False)
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
           
    def __data_generation(self,list_IDs_temp):
        xseq, y = [], []
        for seqid in list_IDs_temp:
            seq = self.Seqs[seqid]
            coding = pre.to_binary(seq)

            coding = pre.repeat_padding(coding,self.length_cutoff)
            xseq.append(coding)
            y.append(self.labels[seqid])
        xseq = np.array(xseq)
        y = np.array(y).reshape([len(y),1])
        return xseq, y
    
class MiniBatch(keras.utils.Sequence):
    # Adapted from https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly
    def __init__(self, IDs_grouped_by_intervals, labels, Seqs, batch_size, weights, propfile=None, shuffle=True,length_cutoff=2000, sample_num_per_epoch=10000,mix_encoding=False):
        self.IDs_grouped_by_intervals      = IDs_grouped_by_intervals   # All IDs in the training datasets
        self.labels        = labels     # a dictonary
        self.batch_size    = batch_size
        self.shuffle       = shuffle
        self.Seqs          = Seqs
        self.length_cutoff = length_cutoff
        self.weights       = weights    # weights for temperature intervals
        self.propfile      = propfile
        self.sample_num_per_epoch = sample_num_per_epoch
        self.mix_encoding  = mix_encoding
        if propfile is not None: 
            self.pro_code = pre.load_aa_properties(propfile)
        
        self.minibatch = self.__sample_a_minibatch()
        
        self.on_epoch_end()
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.sample_num_per_epoch/ self.batch_size))
    
    def __getitem__(self, index):
        'Generate one batch of data'

        # Generate data
        list_IDs_temp = self.minibatch[index*self.batch_size:(index+1)*self.batch_size]
        X, y = self.__data_generation(list_IDs_temp)

        return X, y
    
    def __sample_a_minibatch(self):
        # ogt_db: ogt_db = {(0,5):[recs, list]}
        # w_intervals = {(0, 5): 0.0010515247108307045, (5, 10): 0.0005257623554153522,...}
        # the sampling was done without replacement
        size = self.sample_num_per_epoch # firstly sample 1000, then randomly choose mbatch samples
        
        sampled_sequences = list()
        for intv, w in self.weights.items():
            num = int(np.round(w*size,decimals=0))
            lst_all = self.IDs_grouped_by_intervals.get(intv,[])
            if len(lst_all) <1: continue
            sampled_sequences.extend(np.random.choice(lst_all,size=num))

        np.random.shuffle(sampled_sequences) # random shuffule the results
        return list(sampled_sequences)
    
    def on_epoch_end(self):
        'Updates samples after each epoch'
        self.minibatch = self.__sample_a_minibatch()
           
    
    def __data_generation(self,list_IDs_temp):
        xseq, y = [], []
        for seqid in list_IDs_temp:
            seq = self.Seqs[seqid]
            
            if self.mix_encoding:
                onehot = pre.to_binary(seq)
                prop    = pre.to_properties(seq,self.pro_code)
                coding  = np.hstack((onehot,prop)) 

            elif self.propfile is None: coding = pre.to_binary(seq)
            else: coding = pre.to_properties(seq,self.pro_code)
            
            coding = pre.zero_padding(coding,self.length_cutoff)
            xseq.append(coding)
            y.append(self.labels[seqid])
        xseq = np.array(xseq)
        y = np.array(y).reshape([len(y),1])
        return xseq, y