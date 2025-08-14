import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Normalizer, MinMaxScaler, MaxAbsScaler


def generate_features_labels(file_path,scaler):
    # with open(file_path[0], 'r') as file:
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
    Let's make volumetric data with LES
    '''
    volume_data = [droples.reshape((3,3,3,5)) for droples in sdropdatales]
    '''
    Let's also obtain the number of actual droplets in every cell
    '''
    nactdrops = [drop[3:-1:3].astype('int') for drop in sdrops]
    nactdrops_scaled = scaler.fit_transform(nactdrops)
    channel_1 = [actdrops.reshape((3,3,3)) for actdrops in nactdrops_scaled]
    
    vd = np.array(volume_data)
    print(vd.shape)
    vd_reshaped = vd.reshape((-1,5))
    les_u = scaler.fit_transform(vd_reshaped[:,2:])
    les_s = scaler.fit_transform(vd_reshaped[:,0,None])
    les_t = scaler.fit_transform(vd_reshaped[:,1,None])
    vd_scaled = np.concatenate((les_s,les_t,les_u),axis=1)
    vd_scaled = vd_scaled.reshape(vd.shape)

    volume_data = np.concatenate((vd_scaled,np.expand_dims(channel_1,axis=-1)),axis=-1)
    '''
    We will now take the sdrop data and Euler variables at sdrop location separately, 
    scale them and then concatenate them.
    '''
    sdrop_1 = sdropdata[:,:-1] # Remove the label (supersaturation value)

    nsdropprops = 5
    sdrop_2 = np.zeros((nsdrops, nsdropprops))
    for i, drop in enumerate(sdrops):
        sdrop_2[i] = np.array([float(item) for item in drop[1].split()])

    # Scale and Concatenate
    sdrop_1_scaled = scaler.fit_transform(sdrop_1) 
    sdrop_2_scaled = scaler.fit_transform(sdrop_2)

    X_mlp = np.concatenate((sdrop_1_scaled,sdrop_2_scaled),axis=1) 

    y = np.array([sdrop[-1] for sdrop in sdropdata])

    return X_mlp, volume_data, y, ssdata

# file_path = ["/home/divyaprakash/cloud_tf_cuda/gendata/outdir/003000/training.txt", 
#              "/home/divyaprakash/cloud_tf_cuda/gendata/outdir/005000/training.txt",
#              "/home/divyaprakash/cloud_tf_cuda/gendata/outdir/007000/training.txt"]


file_path = ["/home/divyaprakash/cloud_tf_cuda/outdir_sk1_nm100_nx32/003000/training.txt",
             "/home/divyaprakash/cloud_tf_cuda/outdir_sk1_nm100_nx32/005000/training.txt",
             "/home/divyaprakash/cloud_tf_cuda/outdir_sk1_nm100_nx32/007000/training.txt"]
             

scaler = StandardScaler()
for i, fname in enumerate(file_path):
    if i > 0:
        features_mlp, features_cnn, labels,_ = generate_features_labels(fname,scaler)
        X_mlp = np.concatenate((X_mlp,features_mlp)) 
        X_cnn = np.concatenate((X_cnn,features_cnn)) 
        y = np.concatenate((y,labels))
    else:
        X_mlp, X_cnn, y,_ = generate_features_labels(fname,scaler)

for i, fname in enumerate(file_path):
    if i > 0:
        _,_,_,ssdata = generate_features_labels(fname,scaler)
        ss = np.concatenate((ss,ssdata)) 
    else:
        _,_,_,ss = generate_features_labels(fname,scaler)

np.save('ssdata.npy',ss)        
np.save('features_min.npy',X_mlp)
# np.save('features_cnn.npy',X_cnn)
np.save('labels.npy',y)