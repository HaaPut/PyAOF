import os
import sys
import vtk
import logging
import numpy as np
import matplotlib.pyplot as plt
from types import SimpleNamespace
from pyaof import compute_aof, compute_sdf

# Set up logging for the DEMO (this is okay to do in a script, not in a library)
logger = logging.getLogger('pyaof_demo')
logger.setLevel(level=logging.DEBUG)
ch = logging.StreamHandler()
ch.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(ch)

def main():
    params = SimpleNamespace()
    params.obj = 'cuboid'
    params.resolution = 1000
    params.normalize = True

    data_folder = '../data'
    obj_path = os.path.join(data_folder, f"{params.obj}.obj")
    
    if not os.path.exists(obj_path):
        logger.critical(f"Object {obj_path} not found")
        sys.exit(-1)

    if obj_path.endswith('obj'):
        reader = vtk.vtkOBJReader()
    elif obj_path.endswith('off'):
        reader = vtk.vtkOFFReader()
        
    reader.SetFileName(obj_path)
    reader.Update()
    mesh = reader.GetOutput()
    
    sdf, _, _, spacing = compute_sdf(
        mesh,
        resolution=params.resolution,
        normalize=params.normalize
    )

    logger.info(f"sdf shape: {sdf.shape} spacing {spacing}, inv spacing = {1/spacing[0]}")
    
    aof_vol = compute_aof(
        sdf,
        hx=spacing[0],
        hy=spacing[1],
        hz=spacing[2],
        tile_size=1024
    )

    aof_scale = 2
    thresh = aof_vol.max() * 0.1
    scaled_aof_vals = aof_scale * aof_vol[aof_vol > 0.1]
    logger.info(f"{len(scaled_aof_vals)} relevant aof values in range {aof_vol.min()}->{aof_vol.max()}")
    clipped = aof_scale*aof_vol.copy()
    nans = np.zeros_like(clipped)
    nans[clipped>1] = 100
    clipped[clipped > 1] = 0
    t = params.resolution // 2
    #vals, vecs, fa = get_tangents(aof)
    fig,ax = plt.subplots(1,2,figsize=(20,10))
    #im = ax[0].imshow(np.rad2deg(np.arcsin(clipped[t])) + nans[t],cmap='tab10')
    im = ax[0].imshow(np.rad2deg(np.arcsin(clipped[:,t,:])) + nans[:,t,:])

    plt.colorbar(im,ax=ax[0])
    scaled_aof_vals = np.random.choice(scaled_aof_vals,100000)
    logger.debug(f"Drawn image for AOF@{t}")
    vals,_, bars = ax[1].hist(np.rad2deg(np.arcsin(scaled_aof_vals[scaled_aof_vals<=1])),
                              bins=100)
    logger.debug(f"Computing Histogram of finite angles")
    vals,_, bars = ax[1].hist([100]*(scaled_aof_vals>1).sum(),
                              bins=1)
    logger.debug(f"Computing Histogram of nan angles")
    #ax[1].bar_label(bars,fontsize=20, color='orange')
    #plt.colorbar(im, ax= ax[1])
    
    plt.savefig("debug.png")
    plt.close()
    logger.debug(f"AOF range: {aof_vol.min()} -> {aof_vol.max()}")   

if __name__ == "__main__":
    main()