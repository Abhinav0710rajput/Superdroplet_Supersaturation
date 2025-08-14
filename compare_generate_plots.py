import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({
    'font.size': 18,          # Default font size
    # 'axes.titlesize': 18,     # Font size of the axes title
    # 'axes.labelsize': 16,     # Font size of the x and y labels
    # 'xtick.labelsize': 12,    # Font size of the x tick labels
    # 'ytick.labelsize': 12,    # Font size of the y tick labels
    # 'legend.fontsize': 14,    # Font size of the legend
    # 'figure.titlesize': 20    # Font size of the figure title
})

# raw = np.load('raw_data.npy')
acts = np.load('y_test.npy')
preds = np.load('predictions.npy')
ids = np.load('idorg_test.npy')
ss = np.load('ssdata.npy')

# Superdroplet radius
srad  = ss[ids][:,2]  
sfil  = ss[ids][:,1]  
seff  = ss[ids][:,0]
spred = preds

# We need to sort these according to superdroplet radius and then plot them
idsort = np.argsort(srad)

plt.scatter(srad[idsort],sfil[idsort], color='r',alpha=0.1,label=r'$S_{fil}$')
plt.scatter(srad[idsort],seff[idsort], color='b',alpha=0.1,label=r'$S_{eff}$')
plt.scatter(srad[idsort],spred[idsort],color='g',alpha=0.1,label=r'$S_{pred}$')

plt.legend()
fig = plt.gcf()
fig.tight_layout()
plt.xlabel('Superdroplet Radius')
plt.ylabel('Super-saturation')
# plt.show()
plt.savefig('images/s3rad.png', dpi=300)
plt.close()

##################### percentile values/bin edges #################################
nquantile = 5
dquantile = 1/nquantile
pvals = np.arange(0,1+dquantile,dquantile)
binedges = np.quantile(srad,pvals)
# binedges = np.insert(binedges,0,0) # insert 0 in the first place
# we extend the last binedge to beyont the maximum value
binedges[-1] = binedges[-1] + binedges[-1]/100 # this is done to keep everything inside the last binedge
idbins = np.digitize(srad,binedges)

# # bin the actual values
# nbinedges = 7
# binedges = np.linspace(-0.9,0.1,nbinedges)
# idbins = np.digitize(acts,binedges)

# # for id in np.unique(idbins):
#     # acts[idbins==id]
    
# #binerror = [np.abs((acts[idbins==id].squeeze()-preds[idbins==id].squeeze())/acts[idbins==id].squeeze())*100 for id in np.unique(idbins)]
binerror = [acts[idbins==id].squeeze()-preds[idbins==id].squeeze() for id in np.unique(idbins)]
binerror2 = [acts[idbins==id].squeeze()-sfil[idbins==id].squeeze() for id in np.unique(idbins)]

long_dash = chr(0x2014)
xlabelstrings = [f"({binedges[i-1]:.2f}){long_dash}({binedges[i]:.2f})\n{sum(idbins==i)}" for i in np.unique(idbins)]
xlabelstrings2 = [f"({binedges[i-1]:.2f}){long_dash}({binedges[i]:.2f})\n{sum(idbins==i)}" for i in np.unique(idbins)]

bp1 = plt.boxplot(binerror,labels=xlabelstrings,showfliers=False, widths=0.6, patch_artist=True,
                  boxprops=dict(facecolor='lightgreen', color='green', alpha=0.6),
                  medianprops=dict(color='green', alpha=0.6),
                  whiskerprops=dict(color='green', alpha=0.6),
                  capprops=dict(color='green', alpha=0.6),
                  flierprops=dict(color='green', markeredgecolor='green', alpha=0.6))
bp2 = plt.boxplot(binerror2,labels=xlabelstrings2,showfliers=False, widths=0.4, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', color='blue', alpha=0.6),
                  medianprops=dict(color='blue', alpha=0.6),
                  whiskerprops=dict(color='blue', alpha=0.6),
                  capprops=dict(color='blue', alpha=0.6),
                  flierprops=dict(color='blue', markeredgecolor='blue', alpha=0.6))
plt.xticks(rotation=0)
plt.legend([bp1["boxes"][0], bp2["boxes"][0]], [r"$S_{eff}-S_{pred}$", r"$S_{eff}-S_{fil}$"], loc='upper right')

# plt.ylabel("ape")
# plt.ylabel("ground truth - prediction")
plt.ylabel("Error")
plt.xlabel(r"Radius bins [$\mu{m}$]")
fig = plt.gcf()
# fig.tight_layout()
# plt.savefig('images/radbins.png', dpi=300)
# plt.show()
# plt.close()

ax = plt.gca()

# inset plot
ax_inset = inset_axes(ax, width="30%", height="30%", loc='lower right', bbox_to_anchor=(0.1, 0.1, 0.8, 0.6), bbox_transform=ax.transAxes)  # adjust the size and location
fig.add_axes(ax_inset)

err = acts[idsort].squeeze() - preds[idsort].squeeze()
# plt.plot(srad[idsort],err, color='r',alpha=0.1,label=r'$s_{fil}$')
# shading quantiles
# redefine quantiles to remove ambiguity from the end
binedges = np.quantile(srad,pvals)
colors = ['lightgray', 'lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
for i in range(nquantile):
    ax_inset.fill_betweenx(y=[min(err), max(err)], x1=binedges[i], x2=binedges[i+1], color=colors[i], alpha=0.7, label=f'q{i+1}')

ax_inset.scatter(srad[idsort],err,s=1,color='r',alpha=1.0,label=r'$s_{fil}$')
ax_inset.vlines(binedges, min(err), max(err), color='black', linestyle='--', linewidth=1)
plt.ylabel("Error")
plt.xlabel(r"Radius [$\mu{m}$]")
fig.set_size_inches(16, 8)  # width: 16 inches, height: 8 inches
fig.tight_layout()
plt.savefig('images/compare_rad_qtiles_bins.png', dpi=300)
plt.close()


##################### percentile values/bin edges #################################
nbinedges = 6
# we extend the last binedge to beyont the maximum value
binedges = np.linspace(0.9999*min(srad),1.00001*max(srad),nbinedges)
idbins = np.digitize(srad,binedges)

# # bin the actual values
# nbinedges = 7
# binedges = np.linspace(-0.9,0.1,nbinedges)
# idbins = np.digitize(acts,binedges)

# # for id in np.unique(idbins):
#     # acts[idbins==id]
    
# #binerror = [np.abs((acts[idbins==id].squeeze()-preds[idbins==id].squeeze())/acts[idbins==id].squeeze())*100 for id in np.unique(idbins)]
binerror = [acts[idbins==id].squeeze()-preds[idbins==id].squeeze() for id in np.unique(idbins)]
binerror2 = [acts[idbins==id].squeeze()-sfil[idbins==id].squeeze() for id in np.unique(idbins)]

long_dash = chr(0x2014)
xlabelstrings = [f"({binedges[i-1]:.2f}){long_dash}({binedges[i]:.2f})\n{sum(idbins==i)}" for i in np.unique(idbins)]
xlabelstrings2 = [f"({binedges[i-1]:.2f}){long_dash}({binedges[i]:.2f})\n{sum(idbins==i)}" for i in np.unique(idbins)]

bp1 = plt.boxplot(binerror,labels=xlabelstrings,showfliers=False, widths=0.6, patch_artist=True,
                  boxprops=dict(facecolor='lightgreen', color='green', alpha=0.6),
                  medianprops=dict(color='green', alpha=0.6),
                  whiskerprops=dict(color='green', alpha=0.6),
                  capprops=dict(color='green', alpha=0.6),
                  flierprops=dict(color='green', markeredgecolor='green', alpha=0.6))
bp2 = plt.boxplot(binerror2,labels=xlabelstrings2,showfliers=False, widths=0.4, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', color='blue', alpha=0.6),
                  medianprops=dict(color='blue', alpha=0.6),
                  whiskerprops=dict(color='blue', alpha=0.6),
                  capprops=dict(color='blue', alpha=0.6),
                  flierprops=dict(color='blue', markeredgecolor='blue', alpha=0.6))
plt.xticks(rotation=0)
plt.legend([bp1["boxes"][0], bp2["boxes"][0]], [r"$S_{eff}-S_{pred}$", r"$S_{eff}-S_{fil}$"], loc='upper right')

# plt.ylabel("ape")
# plt.ylabel("ground truth - prediction")
plt.ylabel("Error")
plt.xlabel(r"Radius bins [$\mu{m}$]")
fig = plt.gcf()
# fig.tight_layout()
# plt.savefig('images/radbins.png', dpi=300)
# plt.show()
# plt.close()

ax = plt.gca()

# inset plot
ax_inset = inset_axes(ax, width="30%", height="30%", loc='lower right', bbox_to_anchor=(0.1, 0.1, 0.8, 0.6), bbox_transform=ax.transAxes)  # adjust the size and location
fig.add_axes(ax_inset)

err = acts[idsort].squeeze() - preds[idsort].squeeze()
# plt.plot(srad[idsort],err, color='r',alpha=0.1,label=r'$s_{fil}$')
# shading quantiles
# redefine quantiles to remove ambiguity from the end
# binedges = np.quantile(srad,pvals)
colors = ['lightgray', 'lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
for i in range(nbinedges-1):
    ax_inset.fill_betweenx(y=[min(err), max(err)], x1=binedges[i], x2=binedges[i+1], color=colors[i], alpha=0.7, label=f'q{i+1}')

ax_inset.scatter(srad[idsort],err,s=1,color='r',alpha=1.0,label=r'$s_{fil}$')
ax_inset.vlines(binedges, min(err), max(err), color='black', linestyle='--', linewidth=1)
plt.ylabel("Error")
plt.xlabel(r"Radius [$\mu{m}$]")
fig.set_size_inches(16, 8)  # width: 16 inches, height: 8 inches
fig.tight_layout()
plt.savefig('images/compare_rad_uniform_bins.png', dpi=300)
plt.close()


