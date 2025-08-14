import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file_path = "/home/divyaprakash/cloud_tf_cuda/gendata/outdir/003000/training.txt"

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
    sdropdata[i] = np.array([float(item) for item in drop[1].split()])

adrops = 0
for i, drop in enumerate(sdrops):
    adrops = adrops + drop[3:-1:3].astype('int').sum()
    
# # Plot the superdroplets
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# file_path = "/home/divyaprakash/cloud_tf_cuda/gendata/outdir/003000/lag_super.txt"
# sdropdata = pd.read_csv(file_path,sep=' ').to_numpy() 
# ax.scatter(sdropdata[:,0],sdropdata[:,1],sdropdata[:,2],c='r',marker='o')
# plt.show()
