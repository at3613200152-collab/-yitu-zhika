"""Render scientific figure in a process without PyTorch/OpenMP runtime mixing."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

output=Path(__file__).resolve().parents[1]/'results/hsi_unet_v2'
data=np.load(output/'test_examples.npz',allow_pickle=False)
fig,axes=plt.subplots(3,4,figsize=(12,9),layout='constrained')
for i,(array,key) in enumerate(zip(data['images'],data['ids'])):
    images=[array[:3].transpose(1,2,0),array[3],array[4],np.abs(array[3]-array[4])]
    for j,image in enumerate(images):
        axes[i,j].imshow(image,cmap='magma' if j==3 else 'gray',vmin=0,vmax=.3 if j==3 else 1)
        axes[i,j].axis('off')
        axes[i,j].set_title(['RGB','Measured NIR (scaled)','Predicted NIR','Absolute error'][j]+('\n'+key if j==0 else ''))
fig.suptitle('HSIFoodIngr held-out date: fixed first 3 scans; CPU float32 visualization')
fig.savefig(output/'test_examples.png',dpi=140)
plt.close(fig)
