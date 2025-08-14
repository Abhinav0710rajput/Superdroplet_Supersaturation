import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D

# raw = np.load('raw_data.npy')
acts = np.load('y_test.npy')
preds = np.load('predictions.npy')

# Bin the actual values
nbinedges = 5
binedges = np.linspace(-0.9,0.1,nbinedges)
idbins = np.digitize(acts,binedges)

# for id in np.unique(idbins):
    # acts[idbins==id]
    
#binerror = [np.abs((acts[idbins==id].squeeze()-preds[idbins==id].squeeze())/acts[idbins==id].squeeze())*100 for id in np.unique(idbins)]
binerror = [acts[idbins==id].squeeze()-preds[idbins==id].squeeze() for id in np.unique(idbins)]

long_dash = chr(0x2014)
xlabelstrings = [f"({binedges[i-1]:.2f}){long_dash}({binedges[i]:.2f})\n{sum(idbins==i)}" for i in np.unique(idbins)]

plt.boxplot(binerror,labels=xlabelstrings,showfliers=False)
plt.xticks(rotation=60)
# plt.ylabel("APE")
# plt.ylabel("Ground Truth - Prediction")
plt.ylabel("Error")
plt.xlabel("Bins")
fig = plt.gcf()
fig.tight_layout()
plt.show()


# '''
# Plot the error wrt to superdroplets location
# '''
# idorg = np.load('idorg_test.npy')

# file_path = ["/home/divyaprakash/cloud_tf_cuda/outdir_sk1_nm300_nx32/003000/lag_super.txt",
#              "/home/divyaprakash/cloud_tf_cuda/outdir_sk1_nm300_nx32/005000/lag_super.txt",
#              "/home/divyaprakash/cloud_tf_cuda/outdir_sk1_nm300_nx32/007000/lag_super.txt"]


# loclist = [pd.read_csv(filename,delimiter=' ').to_numpy() for filename in file_path]
# sdrop_locs = np.concatenate(loclist)

# #ape = np.abs((acts.squeeze()-preds.squeeze())/acts.squeeze())*100
# ape = np.abs(acts.squeeze()-preds.squeeze())


# # Create a 3D scatter plot
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# sc = ax.scatter(sdrop_locs[idorg,0], sdrop_locs[idorg,1], sdrop_locs[idorg,2], c=ape, cmap='jet', alpha=0.2)
# sc.set_clim([0,0.10])
# #sc.set_clim([0,100])
# plt.colorbar(sc)
# plt.show()
