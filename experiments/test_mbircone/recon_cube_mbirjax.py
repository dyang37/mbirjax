import numpy as np
import os
import time
import pprint
import jax.numpy as jnp
import mbirjax
import mbirjax.plot_utils as pu
import from_file_mbircone
from demo_utils import plot_image, nrmse

if __name__ == "__main__":
    """
    This script performs an MBIRJAX recon using a synthetic cubic phantom. This includes:
    1. Generate a cubic phantom in MBIRJAX.
    2. Load the geometry parameters from a yaml file, which is generated from an MBIRCONE reconstruction.
    3. Forward project the phantom using MBIRJAX projector.
    4. Performs reconstruction using MBIRJAX recon.
    
    The geometry parameters are defined in MBIRCONE, and converted to MBIRJAX parameters using the helper function "from_file_mbircone". This is to make sure that the geometry params in MBIRCONE and MBIRJAX are aligned.
    Please run "recon_cube_mbircone.py" to generate the MBIRCONE parameter and data files.
    """
    
    # local path to save phantom, sinogram, and reconstruction images
    save_path = f'output/3D_cube_mbirjax/'
    os.makedirs(save_path, exist_ok=True)

    # recon parameters
    sharpness=0.0

    # Choose the geometry type
    geometry_type = 'cone'  # 'cone' or 'parallel'

    # load MBIRCONE geometry params. Please run "recon_cube_mbircone.py" to generate this file.
    filename_mbircone = "output/3D_cube_mbircone/params_dict_mbircone.yaml"
    # convert MBIRCONE geometry params to MBIRJAX geometry params
    params_dict_mbirjax = from_file_mbircone.from_file_mbircone(filename_mbircone)

    # Set up the model based on MBIRJAX geometry params
    ct_model = mbirjax.ConeBeamModel(sinogram_shape=params_dict_mbirjax["sinogram_shape"], 
                                     angles=params_dict_mbirjax["angles"], 
                                     source_detector_dist=params_dict_mbirjax["source_detector_dist"], 
                                     source_iso_dist=params_dict_mbirjax["source_iso_dist"],
                                     magnification=params_dict_mbirjax["magnification"],
                                     delta_det_channel=params_dict_mbirjax["delta_det_channel"],
                                     delta_det_row=params_dict_mbirjax["delta_det_row"],
                                     delta_voxel=params_dict_mbirjax["delta_voxel"],
                                     det_channel_offset=params_dict_mbirjax["det_channel_offset"],
                                     det_row_offset=params_dict_mbirjax["det_row_offset"],
                                     )

    # Generate 3D Shepp Logan phantom
    print('Creating phantom')
    phantom = ct_model.gen_modified_3d_sl_phantom()
    # Set phantom generation parameters
    num_views, num_det_rows, num_det_channels = params_dict_mbirjax["sinogram_shape"]
    num_phantom_slices = num_det_rows           # Set number of phantom slices = to the number of detector rows
    num_phantom_rows = num_det_channels         # Make number of phantom rows and columns = to number of detector columns
    num_phantom_cols = num_det_channels

    phantom = np.zeros((num_phantom_rows, num_phantom_cols, num_phantom_slices))
    # Set the central cubic region to 0.1
    phantom[num_phantom_rows//4:num_phantom_rows*3//4,
            num_phantom_cols//4:num_phantom_cols*3//4,
            num_phantom_slices//4:num_phantom_cols*3//4] = 0.1
    print('Phantom shape = ', np.shape(phantom))
    pu.slice_viewer(phantom, phantom, title='Phantom axial slice (left) and coronal slice (right)', slice_axis=2, slice_axis2=0)

    # Generate synthetic sinogram data
    print('Creating sinogram')
    sinogram_mbirjax = ct_model.forward_project(phantom)
    
    # load MBIRCONE sinogram. Please run "recon_cube_mbircone.py" to generate this file.
    sinogram_mbircone = np.load("output/3D_cube_mbircone/sino_mbircone.npy")
    print("NRMSE between MBIRCONE sinogram and MBIRJAX sinogram = ", nrmse(sinogram_mbirjax, sinogram_mbircone))
    diff_sinogram = sinogram_mbirjax - sinogram_mbircone


    # View sinogram and diff_sinogram
    pu.slice_viewer(sinogram_mbirjax, sinogram_mbircone, title='Sinogram mbirjax (left) vs Sinogram mbircone (right)', slice_axis=0)
    pu.slice_viewer(diff_sinogram, title='Sinogram differene (sino_mbirjax - sino_mbircone)', slice_axis=0, vmin=-4.5, vmax=4.5)
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
    recon, recon_params = ct_model.recon(sinogram_mbirjax, weights=weights)

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

    # Display results
    pu.slice_viewer(phantom, recon, title='Phantom (left) vs VCD Recon (right)')

