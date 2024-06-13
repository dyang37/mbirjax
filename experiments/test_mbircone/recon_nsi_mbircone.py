import os, sys
import numpy as np
import math
import urllib.request
import tarfile
import mbircone
import to_file_mbircone
import demo_utils
import pprint
pp = pprint.PrettyPrinter(indent=4)

# ###########################################################################
# Set the parameters to get the data and do the recon 
# ###########################################################################

# ###################### User defined params. Change the parameters below for your own use case.
save_path = './output/pipettor_vert_ds2_alu_pitch' # path to store output recon images
os.makedirs(save_path, exist_ok=True) # mkdir if directory does not exist
downsample_factor = [2, 2] # downsample factor of scan images along detector rows and detector columns.
subsample_view_factor = 2

# path to NSI dataset
dataset_path = "/depot/bouman/data/share_conebeam_data/Pipettor/pipettor_vertical"

# ###################### NSI specific file paths, These are derived from dataset_path.
# User may change the variables below for a different NSI dataset.
# path to NSI config file. Change dataset path params for your own NSI dataset
nsi_config_file_path = os.path.join(dataset_path, 'Ovation_Vertical.nsipro')
# path to "Geometry Report.rtf"
geom_report_path = os.path.join(dataset_path, 'Geometry Report [Geometry].rtf')
# path to directory containing all object scans
obj_scan_path = os.path.join(dataset_path, 'Radiographs')
# path to blank scan. Usually <dataset_path>/Corrections/gain0.tif
blank_scan_path = os.path.join(dataset_path, 'Corrections/gain0.tif')
# path to dark scan. Usually <dataset_path>/Corrections/offset.tif
dark_scan_path = os.path.join(dataset_path, 'Corrections/offset.tif')
# path to NSI file containing defective pixel information
defective_pixel_path = os.path.join(dataset_path, 'Corrections/defective_pixels.defect')
# ###################### End of parameters

# ###########################################################################
# NSI preprocess: obtain sinogram, sino weights, angles, and geometry params
# ###########################################################################
print("\n********************************************************************************",
      "\n** Load scan images, angles, geometry params, and defective pixel information **",
      "\n********************************************************************************")
obj_scan, blank_scan, dark_scan, angles, geo_params, defective_pixel_list = \
        mbircone.preprocess.NSI_load_scans_and_params(nsi_config_file_path, geom_report_path,
                                                      obj_scan_path, blank_scan_path, dark_scan_path,
                                                      defective_pixel_path,
                                                      downsample_factor=downsample_factor,
                                                      subsample_view_factor=subsample_view_factor)

print("MBIR geometry paramemters:")
pp.pprint(geo_params)
print('obj_scan shape = ', obj_scan.shape)
print('blank_scan shape = ', blank_scan.shape)
print('dark_scan shape = ', dark_scan.shape)

# extract mbircone geometry params required for recon
dist_source_detector = geo_params["dist_source_detector"]
magnification = geo_params["magnification"]
delta_det_row = geo_params["delta_det_row"]
delta_det_channel = geo_params["delta_det_channel"]
det_channel_offset = geo_params["det_channel_offset"]
det_row_offset = geo_params["det_row_offset"]
rotation_offset = geo_params["rotation_offset"]
num_det_rows = geo_params["num_det_rows"]
num_det_channels = geo_params["num_det_channels"]

# convert 1 ALU = 1 detector pixel pitch
dist_source_detector /= delta_det_channel
det_channel_offset /= delta_det_channel
det_row_offset /= delta_det_row
rotation_offset /= delta_det_channel
delta_det_channel = 1.0
delta_det_row = 1.0

filename = os.path.join(save_path, "params_dict_mbircone.yaml")

to_file_mbircone.to_file_mbircone(filename,
                                  angles, num_det_rows, num_det_channels, dist_source_detector, magnification,
                                  delta_det_channel=delta_det_channel, delta_det_row=delta_det_row,
                                  det_channel_offset=det_channel_offset, det_row_offset=det_row_offset, rotation_offset=rotation_offset)


print("\n*******************************************************",
      "\n********** Compute sinogram from scan images **********",
      "\n*******************************************************")
sino, defective_pixel_list = \
        mbircone.preprocess.transmission_CT_compute_sino(obj_scan, blank_scan, dark_scan,
                                                         defective_pixel_list
                                                        )
# delete scan images to optimize memory usage
del obj_scan, blank_scan, dark_scan

print("\n*******************************************************",
      "\n********* Interpolate defective sino entries **********",
      "\n*******************************************************")
sino, defective_pixel_list = mbircone.preprocess.interpolate_defective_pixels(sino, defective_pixel_list)

print("\n*******************************************************",
      "\n************** Correct background offset **************",
      "\n*******************************************************")
background_offset = mbircone.preprocess.calc_background_offset(sino)
print("background_offset = ", background_offset)
sino = sino - background_offset

print("\n*******************************************************",
      "\n**** Rotate sino images w.r.t. rotation axis tilt *****",
      "\n*******************************************************")
sino = mbircone.preprocess.correct_tilt(sino, tilt_angle=geo_params["rotation_axis_tilt"])

print("\n*******************************************************",
      "\n************** Calculate sinogram weight **************",
      "\n*******************************************************")
weights = mbircone.preprocess.calc_weights(sino, weight_type="transmission_root",
                                           defective_pixel_list=defective_pixel_list
                                          )
np.save(os.path.join(save_path, "sino.npy"), sino)
np.save(os.path.join(save_path, "weights.npy"), weights)

# ###########################################################################
# Perform MBIR reconstruction
# ###########################################################################
print("\n*******************************************************",
      "\n************* Perform MBIR reconstruction *************",
      "\n**** This step will take 30-60 minutes to finish ******",
      "\n*******************************************************")
# MBIR recon
recon_mbir = mbircone.cone3D.recon(sino, angles, dist_source_detector, magnification,
                                   det_channel_offset=det_channel_offset, det_row_offset=det_row_offset,
                                   rotation_offset=rotation_offset,
                                   delta_det_row=delta_det_row, delta_det_channel=delta_det_channel,
                                   weights=weights)
np.save(os.path.join(save_path, "recon_mbir.npy"), recon_mbir)

print("MBIR recon finished. recon shape = ", np.shape(recon_mbir))

input("press Enter")
