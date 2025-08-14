import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, MaxAbsScaler


def generate_features_labels(file_path,scaler):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    lines = [line.strip() for line in lines]    

    lines_per_sdrop = 83
    nsdrops = len(lines)//lines_per_sdrop
    sdrops = np.array_split(lines,nsdrops)

    '''
    Let's first obtain the data of all the super droplets and
    store them in a numpy array
    '''
    nsdropprops = 5
    sdropdata = np.zeros((nsdrops, nsdropprops))
    for i, drop in enumerate(sdrops):
        sdropdata[i] = np.array([float(item) for item in drop[0].split()])

    '''
    Get the filetereS and S
    '''
    ssdata = np.zeros((nsdrops, 2))
    for i, drop in enumerate(sdrops):
        ssdata[i,0] = [float(item) for item in drop[0].split()][-1]
        ssdata[i,1] = [float(item) for item in drop[1].split()][0]

    
    '''
    We'll separate out the LES cell data
    '''
    lesdata = [sdrop[2:-1:3] for sdrop in sdrops]

    nlesprops = 135 # 3*3*3*5
    sdropdatales = np.zeros((nsdrops, nlesprops))
    for i, drop in enumerate(lesdata):
        sdropdatales[i] = np.array([drop[i].split() for i in range(len(drop))]).flatten().astype('float32')
        
    '''
    Let's also obtain the number of actual droplets in every cell
    '''
    nactdrops = [drop[3:-1:3].astype('int') for drop in sdrops]
    
    '''
    Let's concatenate the data of the sdrop and lesdata at that les cells surrounding the sdrop and
    also the number of actual droplets per cell. But before we do so, we will transform and scale the
    data in each of these subsets
    '''
   
    # Feature scaling using a Scaler
    sdropdata_scaled = scaler.fit_transform(sdropdata[:,:-1])
    sdropdatales_scaled = scaler.fit_transform(sdropdatales)
    nactdrops_scaled = scaler.fit_transform(nactdrops)

    # X = np.array([np.concatenate((sdropdata[i,:-1],sdropdatales[i,:],nactdrops[i])) for i in range(len(sdrops))])
    # X = np.array([np.concatenate((sdropdata_scaled[i],sdropdatales_scaled[i,:],nactdrops_scaled[i])) 
    #                 for i in range(len(sdrops))])
    X = np.array([np.concatenate((sdropdata_scaled[i],sdropdatales_scaled[i,:])) for i in range(len(sdrops))])
    y = np.array([sdrop[-1] for sdrop in sdropdata])
    
    return X, y, ssdata

# file_path = ["/home/divyaprakash/cloud_tf_cuda/gendata/outdir/003000/training.txt", 
#              "/home/divyaprakash/cloud_tf_cuda/gendata/outdir/005000/training.txt",
#              "/home/divyaprakash/cloud_tf_cuda/gendata/outdir/007000/training.txt"]


file_path = ["/home/divyaprakash/cloud_tf_cuda/outdir_sk1_nm100_nx32/003000/training.txt",
             "/home/divyaprakash/cloud_tf_cuda/outdir_sk1_nm100_nx32/005000/training.txt",
             "/home/divyaprakash/cloud_tf_cuda/outdir_sk1_nm100_nx32/007000/training.txt"]
             

# scaler = MaxAbsScaler()
scaler = StandardScaler()
for i, fname in enumerate(file_path):
    if i > 0:
        features, labels,_ = generate_features_labels(fname,scaler)
        X = np.concatenate((X,features)) 
        y = np.concatenate((y,labels))
    else:
        X, y,_ = generate_features_labels(fname,scaler)

for i, fname in enumerate(file_path):
    if i > 0:
        _,_,ssdata = generate_features_labels(fname,scaler)
        ss = np.concatenate((ss,ssdata)) 
    else:
        _,_,ss = generate_features_labels(fname,scaler)

np.save('ssdata.npy',ss)        
np.save('features.npy',X)
np.save('labels.npy',y)