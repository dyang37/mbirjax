import numpy as np
import os
import time
import pprint
import jax.numpy as jnp
import mbirjax
import mbirjax.plot_utils as pu
from exp_utils import nrmse
import from_file_mbircone

if __name__ == "__main__":
    """
    This scripts perform an MBIRJAX reconstruction using pre-computed sinogram, weights, and geometry parameters from an MBIRCONE experiment.
    """
    
    # local path to save phantom, sinogram, and reconstruction images
    save_path = f'output/nsi_datasets/autoinjector_vert_ds2_alu_pitch_jax' # change this for different datasets
    os.makedirs(save_path, exist_ok=True)
    
    # path to the MBIRCONE experiment folder. Change this for different datasets.
    save_path_mbircone = "/depot/bouman/users/yang1467/mbircone/demo/output/autoinjector_vert_ds2_alu_pitch"
    # recon parameters
    sharpness=0.0
    snr_db = 40.0

    # Choose the geometry type
    geometry_type = 'cone'  # 'cone' or 'parallel'

    # load MBIRCONE geometry params. Please run "recon_cube_mbircone.py" to generate this file.
    filename_mbircone = os.path.join(save_path_mbircone, "params_dict_mbircone.yaml")
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

       
    # load sinogram and sino weights
    sino = np.load(os.path.join(save_path_mbircone, "sino.npy"))
    weights = np.load(os.path.join(save_path_mbircone, "weights.npy"))

    # Set reconstruction parameter values
    ct_model.set_params(sharpness=sharpness, snr_db=snr_db, verbose=1)

    # Print out model parameters
    ct_model.print_params()
    # ##########################
    # Perform VCD reconstruction
    time0 = time.time()
    recon_jax, recon_params = ct_model.recon(sino, weights=weights, num_iterations=30)

    recon_jax.block_until_ready()
    elapsed = time.time() - time0
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))
    # ##########################

    # Print out parameters used in recon
    pprint.pprint(recon_params._asdict())
    
    np.save(os.path.join(save_path, "recon_jax.npy"), recon_jax)
    #recon_jax = np.load(os.path.join(save_path, "recon_jax.npy"))

    # change shape to the convention of mbircone
    recon_jax = np.transpose(recon_jax, (2,1,0))
    recon_jax = recon_jax[:,:,::-1]
   
    recon_mbircone = np.load(os.path.join(save_path_mbircone, "recon_mbir.npy")) 
    print("NRMSE(recon_jax, recon_mbircone) = ", nrmse(recon_jax, recon_mbircone))
    # Display results
    pu.slice_viewer(recon_mbircone, recon_jax, vmin=0, vmax=0.015, slice_axis=0, slice_label='Axial Slice', title='MBIRCONE recon (left) vs MBIRJAX recon (right)')
    pu.slice_viewer(recon_mbircone, recon_jax, vmin=0, vmax=0.015, slice_axis=1, slice_label='Coronal Slice', title='MBIRCONE recon (left) vs MBIRJAX recon (right)')
    pu.slice_viewer(recon_mbircone, recon_jax, vmin=0, vmax=0.015, slice_axis=2, slice_label='Sagittal Slice', title='MBIRCONE recon (left) vs MBIRJAX recon (right)')

