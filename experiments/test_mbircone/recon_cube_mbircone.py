import os
import numpy as np
import mbircone
from demo_utils import plot_image
import to_file_mbircone

"""
This script performs the following steps: 
1. generate a cubic phantom, 
2. forward project it using mbircone.cone3D.project, 
3. Save the geometry parameters into a yaml file using `to_file_mbircone()`.
4. performs a reconstructon using mbircone.cone3D.recon.
"""

# ###########################################################################
# Set the parameters to generate the phantom, synthetic sinogram, and do the recon
# ###########################################################################

############## MBIRCONE geometry parameters
num_det_rows = 64                           # Number of detector rows
num_det_channels = 64                       # Number of detector channels
dist_source_detector = 4.0*num_det_channels  # Distance from source to detector in ALU
magnification = 1.0                          # Ratio of (source to detector)/(source to center of rotation)
num_views = 32                               # Number of projection views

# Generate uniformly spaced view angles in the range [0, 2*pi).
angles = np.linspace(0, 2 * np.pi, num_views, endpoint=False)

############### recon params
sharpness = 0.0                             # Controls regularization: larger => sharper; smaller => smoother
snr_db = 30.0

# Set phantom generation parameters
num_phantom_slices = num_det_rows           # Set number of phantom slices = to the number of detector rows
num_phantom_rows = num_det_channels         # Make number of phantom rows and columns = to number of detector columns
num_phantom_cols = num_det_channels

# local path to save phantom, sinogram, and reconstruction images
save_path = f'output/3D_cube_mbircone/'
os.makedirs(save_path, exist_ok=True)

print('Genrating a cubic phantom ...\n')
######################################################################################
# Generate a cubic phantom.
######################################################################################
phantom = np.zeros((num_phantom_slices, num_phantom_rows, num_phantom_cols))
# Set the central cubic region to 0.1
phantom[num_phantom_slices//4:num_phantom_slices*3//4,
        num_phantom_rows//4:num_phantom_rows*3//4,
        num_phantom_cols//4:num_phantom_cols*3//4] = 0.1
print('Phantom shape = ', np.shape(phantom))

######################################################################################
# Generate synthetic sinogram
######################################################################################
print('Generating synthetic sinogram ...\n')
sino = mbircone.cone3D.project(phantom, angles,
                               num_det_rows, num_det_channels,
                               dist_source_detector, magnification)
print('Synthetic sinogram shape: (num_views, num_det_rows, num_det_channels) = ', sino.shape)
np.save(os.path.join(save_path, "sino_mbircone.npy"), sino)

######################################################################################
# Save MBIRCONE geometry params into a pickle file
######################################################################################
filename = os.path.join(save_path, "params_dict_mbircone.yaml")
print(f"writing MBIRCONE parameters into {filename} ...\n")
to_file_mbircone.to_file_mbircone(filename,
                                  angles, num_det_rows, num_det_channels, dist_source_detector, magnification)

######################################################################################
# Perform 3D MBIR reconstruction using qGGMRF prior
######################################################################################
print('Performing 3D qGGMRF reconstruction ...\n')
recon = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification, sharpness=sharpness, snr_db=snr_db)
(num_image_slices, num_image_rows, num_image_cols) = np.shape(recon)
print('recon shape = ', np.shape(recon))

# save recon data to disk
np.save(os.path.join(save_path, "recon.npy"), recon)

######################################################################################
# Display phantom, synthetic sinogram, and reconstruction images
######################################################################################
# Set display indexes for phantom and recon images
display_slice_phantom = num_phantom_slices // 2
display_x_phantom = num_phantom_rows // 2
display_y_phantom = num_phantom_cols // 2
display_slice_recon = num_image_slices // 2
display_x_recon = num_image_rows // 2
display_y_recon = num_image_cols // 2

# sinogram images
for view_idx in [0, num_views//4, num_views//2]:
    view_angle = int(angles[view_idx]*180/np.pi)
    plot_image(sino[view_idx, :, :], title=f'sinogram view angle {view_angle} ',
               filename=os.path.join(save_path, f'sino-shepp-logan-3D-view_angle{view_angle}.png'))

vmin = 0
vmax = 0.2

# display phantom images
plot_image(phantom[display_slice_phantom], title=f'phantom, axial slice {display_slice_phantom}',
           filename=os.path.join(save_path, 'phantom_axial.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,display_x_phantom,:], title=f'phantom, coronal slice {display_x_phantom}',
           filename=os.path.join(save_path, 'phantom_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(phantom[:,:,display_y_phantom], title=f'phantom, sagittal slice {display_y_phantom}',
           filename=os.path.join(save_path, 'phantom_sagittal.png'), vmin=vmin, vmax=vmax)
           
# display recon images
plot_image(recon[display_slice_recon], title=f'qGGMRF recon, axial slice {display_slice_recon}',
           filename=os.path.join(save_path, 'recon_axial.png'), vmin=vmin, vmax=vmax)
plot_image(recon[:,display_x_recon,:], title=f'qGGMRF recon, coronal slice {display_x_recon}',
           filename=os.path.join(save_path, 'recon_coronal.png'), vmin=vmin, vmax=vmax)
plot_image(recon[:,:,display_y_recon], title=f'qGGMRF recon, sagittal slice {display_y_recon}',
           filename=os.path.join(save_path, 'recon_sagittal.png'), vmin=vmin, vmax=vmax)
           
print(f"Images saved to {save_path}.") 
input("Press Enter")

