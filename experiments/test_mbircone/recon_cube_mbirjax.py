import numpy as np
import os
import time
import pprint
import jax.numpy as jnp
import mbirjax
import mbirjax.plot_utils as pu
from demo_utils import plot_image, nrmse

if __name__ == "__main__":
    """
    This script performs an MBIRJAX recon using a synthetic cubic phantom. This includes:
    1. Generate a cubic phantom in MBIRJAX.
    2. Forward project the phantom using MBIRJAX projector.
    3. Performs reconstruction using MBIRJAX recon.
    """
    
    # local path to save phantom, sinogram, and reconstruction images
    save_path = f'output/3D_cube_mbirjax/'
    os.makedirs(save_path, exist_ok=True)

    # Choose the geometry type
    geometry_type = 'cone'  # 'cone' or 'parallel'

    # Set parameters
    num_views = 32
    num_det_rows = 64
    num_det_channels = 64
    sharpness = 0.0

    # These can be adjusted to describe the geometry in the cone beam case.
    # np.Inf is an allowable value, in which case this is essentially parallel beam
    source_detector_dist = 4 * num_det_channels
    source_iso_dist = source_detector_dist

    #detector_cone_angle = 2 * np.arctan2(num_det_channels / 2, source_detector_dist)

    # view angles are equally spaced in the range from 0 to 2pi.
    start_angle = 0
    end_angle = 2*np.pi

    # Initialize sinogram
    sinogram_shape = (num_views, num_det_rows, num_det_channels)
    angles = jnp.linspace(start_angle, end_angle, num_views, endpoint=False)

    # Set up the model based on MBIRJAX geometry params
    ct_model = mbirjax.ConeBeamModel(sinogram_shape, angles, source_detector_dist=source_detector_dist, source_iso_dist=source_iso_dist)
    # Generate 3D Shepp Logan phantom
    # Set phantom generation parameters
    num_phantom_slices = num_det_rows           # Set number of phantom slices = to the number of detector rows
    num_phantom_rows = num_det_channels         # Make number of phantom rows and columns = to number of detector columns
    num_phantom_cols = num_det_channels

    print('Creating a rectangular phantom')
    phantom = np.zeros((num_phantom_rows, num_phantom_cols, num_phantom_slices))
    # Set the central cubic region to 0.1
    phantom[num_phantom_rows//4:num_phantom_rows*3//4,
            num_phantom_cols//4:num_phantom_cols*3//4,
            num_phantom_slices//4:num_phantom_cols*3//4] = 0.1
    print('Phantom shape = ', np.shape(phantom))

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram = ct_model.forward_project(phantom)
    # View sinogram
    pu.slice_viewer(sinogram, title='Original sinogram', slice_axis=0, slice_label='View')    
    
    # Generate weights array - for an initial reconstruction, use weights = None, then modify as desired.
    weights = None
    # weights = ct_model.gen_weights(sinogram / sinogram.max(), weight_type='transmission_root')

    # Set reconstruction parameter values
    ct_model.set_params(sharpness=sharpness, verbose=1)

    # Print out model parameters
    ct_model.print_params()

    # ##########################
    # Perform VCD reconstruction
    time0 = time.time()
    recon, recon_params = ct_model.recon(sinogram, weights=weights)

    recon.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))
    # ##########################

    # Print out parameters used in recon
    pprint.pprint(recon_params._asdict())

    max_diff = np.amax(np.abs(phantom - recon))
    print('Geometry = {}'.format(geometry_type))
    nrmse = np.linalg.norm(recon - phantom) / np.linalg.norm(phantom)
    pct_95 = np.percentile(np.abs(recon - phantom), 95)
    print('NRMSE between recon and phantom = {}'.format(nrmse))
    print('Maximum pixel difference between phantom and recon = {}'.format(max_diff))
    print('95% of recon pixels are within {} of phantom'.format(pct_95))

    # change the image data shape to (slices, rows, cols), so that the rotation axis points up when viewing the coronal slices with slice_viewer.
    recon = np.transpose(recon, (2,1,0))
    recon = recon[:,:,::-1] # top should be the 0th slice
    
    phantom = np.transpose(phantom, (2,1,0))
    phantom = phantom[:,:,::-1] # top should be the 0th slice

    # Display results
    pu.slice_viewer(phantom, recon, title='Phantom (left) vs VCD Recon (right)', slice_axis=0, slice_label='Axial Slice', vmin=0, vmax=0.2)
    pu.slice_viewer(phantom, recon, title='Phantom (left) vs VCD Recon (right)', slice_axis=1, slice_label='Coronal Slice', vmin=0, vmax=0.2)

