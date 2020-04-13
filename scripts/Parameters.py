
def p_RES1_uniDist():
    # The optimized parameters, with RES1 archetechture, on a uniform distributed OGT dataset
    params = {
        # intial conv
        'filters': 512,    # 128, ok
        'kernel_size1': 9, # 1, ok
        # kernel size of (6 and 10) are close to the average length of beta-sheet and alpha-helix in a protein
        # knowing the full distribution of amino acids that occur at a position (and in its vicinity, 
        # typically ~7 residues on either side) throughout evolution provides a much better picture of the structural 
        # tendencies near that position
        
        # res1
        'kernel_size21':  21,   # 1, ok
        'kernel_size22':  11,   # 3, ok
        'dilation2': 1,        # 2, ok
        
        
        # Pooling
        'pool_size1': 50,  # 3
        
        # dense 1
        'dense1': 512,    # 256 
        
        # dropout 1
        'dropout1': 0.17,

        # dense 2
        'dense2': 512,     # old 64
        
        # dropout 2
        'dropout2': 0.15,
        
        'lr': 0.0001,
        'mbatch': 32,
        'patience': 50, 
        'min_delta': 0.01,
        'epochs': 500,
        'res_num': 1,
        
    }
    
    return params

def p_RES1_uniDist_best1():
    # The optimized parameters, with RES1 archetechture, on a uniform distributed OGT dataset
    params = {
        # intial conv
        'filters': 512,    # 128, ok
        'kernel_size1': 9, # 1, ok
        # kernel size of (6 and 10) are close to the average length of beta-sheet and alpha-helix in a protein
        # knowing the full distribution of amino acids that occur at a position (and in its vicinity, 
        # typically ~7 residues on either side) throughout evolution provides a much better picture of the structural 
        # tendencies near that position
        
        # res1
        'kernel_size21':  21,   # 1, ok
        'kernel_size22':  11,   # 3, ok
        'dilation2': 1,        # 2, ok
        
        
        # Pooling
        'pool_size1': 50,  # 3
        
        # dense 1
        'dense1': 512,    # 256 
        
        # dropout 1
        'dropout1': 0.17,

        # dense 2
        'dense2': 512,     # old 64
        
        # dropout 2
        'dropout2': 0.15,
        
        'lr': 0.0001,
        'mbatch': 128,
        'patience': 200, 
        'min_delta': 0.01,
        'epochs': 500,
        'res_num': 1,
        
    }
    
    return params


def p_RES2_oriDist():
    # The optimized parameters, with RES2 archetechture, on a originally distributed OGT dataset
    params = {
        # intial conv
        'filters': 512,    # 128, ok
        'kernel_size1': 7, # 1, ok
        # kernel size of (6 and 10) are close to the average length of beta-sheet and alpha-helix in a protein
        # knowing the full distribution of amino acids that occur at a position (and in its vicinity, 
        # typically ~7 residues on either side) throughout evolution provides a much better picture of the structural 
        # tendencies near that position
        
        # res1
        'kernel_size21':  7,   # 1, ok
        'kernel_size22':  7,   # 3, ok
        'dilation2': 1,        # 2, ok
        
         # res2
        'kernel_size31':  31,   # 1, ok
        'kernel_size32':  21,   # 3, ok
        'dilation3': 3,        # 2, ok
        
        
        # Pooling
        'pool_size1': 30,  # 3
        
        # dense 1
        'dense1': 512,    # 256 
        
        # dropout 1
        'dropout1': 0.35,

        # dense 2
        'dense2': 512,     # 256
        
        # dropout 2
        'dropout2': 0.37,
        
        'lr': 0.0005,
        'mbatch': 32,
        'patience': 50, 
        'min_delta': 0.01,
        'epochs': 500,
        'res_num': 2,
        
    }
    
    return params

def p_ResNetRed_oriDist():
    # The optimized parameters, with ResNetRed archetechture, on a dataset with original OGT distribution
    params = {
        # intial conv
        'filters': 512,    # 128, ok
        'kernel_size1': 11, # 1, ok
        # kernel size of (6 and 10) are close to the average length of beta-sheet and alpha-helix in a protein
        # knowing the full distribution of amino acids that occur at a position (and in its vicinity, 
        # typically ~7 residues on either side) throughout evolution provides a much better picture of the structural 
        # tendencies near that position
        
        'pool_type_1': 'avg',
        'pool_size1': 4, 
        
        # res1
        'kernel_size21':  11,   # 1, ok
        'kernel_size22':  7,   # 3, ok
        'dilation2': 2,        # 2, ok
        
        # conv1d with stride
        'kernel_size23': 21,  
        'stride_23':     4,
        
        # res2
        'kernel_size31':  11,   # 1, ok
        'kernel_size32':  31,   # 3, ok
        'dilation3': 5,                 # 3, ok
        
        # conv1d with stride
        'kernel_size33': 31,  
        'stride_33':     3,
        
        # res3
        'kernel_size41':  11,   # 1, ok
        'kernel_size42':  3,   # 3, ok
        'dilation4': 2,   
        
        # Pooling
        'pool_type_2': 'max',
        'pool_size2': 2,
        
        # dense 1
        'dense1': 512,    # 256 
        
        # dropout 1
        'dropout1': 0.12,

        # dense 2
        'dense2': 512,     # old 128
        
        # dropout 2
        'dropout2': 0.43,
        
        'lr': 0.001,
        'mbatch': 32,
        'patience': 50, 
        'min_delta': 0.01,
        'epochs': 500,
        'res_num': 3,
        
    }
    
    return params

def p_ResNetRed_uniDist():
    # The optimized parameters, with ResNetRed archetechture, on a dataset with uniform OGT distribution
    params = {
        # intial conv
        'filters': 512,    # 128, ok
        'kernel_size1': 21, # 1, ok
        # kernel size of (6 and 10) are close to the average length of beta-sheet and alpha-helix in a protein
        # knowing the full distribution of amino acids that occur at a position (and in its vicinity, 
        # typically ~7 residues on either side) throughout evolution provides a much better picture of the structural 
        # tendencies near that position
        
        'pool_type_1': 'max',
        'pool_size1': 2, 
        
        # res1
        'kernel_size21':  9,   # 1, ok
        'kernel_size22':  7,   # 3, ok
        'dilation2': 5,        # 2, ok
        
        # conv1d with stride
        'kernel_size23': 9,  
        'stride_23':     3,
        
        # res2
        'kernel_size31':  3,   # 1, ok
        'kernel_size32':  11,   # 3, ok
        'dilation3': 1,                 # 3, ok
        
        # conv1d with stride
        'kernel_size33': 21,  
        'stride_33':     2,
        
        # res3
        'kernel_size41':  3,   # 1, ok
        'kernel_size42':  21,   # 3, ok
        'dilation4': 3,   
        
        # Pooling
        'pool_type_2': 'max',
        'pool_size2': 2,
        
        # dense 1
        'dense1': 512,    # 256 
        
        # dropout 1
        'dropout1': 0.25,

        # dense 2
        'dense2': 512,     # old 256
        
        # dropout 2
        'dropout2': 0.09,
        
        'lr': 0.0005,
        'mbatch': 32,
        'patience': 50, 
        'min_delta': 0.01,
        'epochs': 500,
        'res_num': 3,
        
    }
    
    return params

def p_ResNetRed_uniDist_best1():
    # The optimized parameters, with ResNetRed archetechture, on a dataset with uniform OGT distribution
    params = {
        # intial conv
        'filters': 512,    # 128, ok
        'kernel_size1': 21, # 1, ok
        # kernel size of (6 and 10) are close to the average length of beta-sheet and alpha-helix in a protein
        # knowing the full distribution of amino acids that occur at a position (and in its vicinity, 
        # typically ~7 residues on either side) throughout evolution provides a much better picture of the structural 
        # tendencies near that position
        
        'pool_type_1': 'max',
        'pool_size1': 2, 
        
        # res1
        'kernel_size21':  9,   # 1, ok
        'kernel_size22':  7,   # 3, ok
        'dilation2': 5,        # 2, ok
        
        # conv1d with stride
        'kernel_size23': 9,  
        'stride_23':     3,
        
        # res2
        'kernel_size31':  3,   # 1, ok
        'kernel_size32':  11,   # 3, ok
        'dilation3': 1,                 # 3, ok
        
        # conv1d with stride
        'kernel_size33': 21,  
        'stride_33':     2,
        
        # res3
        'kernel_size41':  3,   # 1, ok
        'kernel_size42':  21,   # 3, ok
        'dilation4': 3,   
        
        # Pooling
        'pool_type_2': 'max',
        'pool_size2': 2,
        
        # dense 1
        'dense1': 512,    # 256 
        
        # dropout 1
        'dropout1': 0.25,

        # dense 2
        'dense2': 512,     # old 256
        
        # dropout 2
        'dropout2': 0.09,
        
        'lr': 0.0001,
        'mbatch': 128,
        'patience': 50, 
        'min_delta': 0.01,
        'epochs': 500,
        'res_num': 3,
        
    }
    
    return params
