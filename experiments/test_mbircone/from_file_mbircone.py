from ruamel.yaml import YAML
from mbirjax import ParameterHandler
import jax.numpy as jnp
import numpy as np
import copy
import pprint
pp = pprint.PrettyPrinter(indent=4)

def from_file_mbircone(filename):
    """ This function loads MBIRCONE parameters from a yaml file saved using `to_file_mbircone()`, convert the MBIRCONE parameters into MBIRJAX parameters, and return a dictionary which contains the MBIRJAX parameters.

    Args:
        filename: filename (str): Name of the file containing parameters to load. Must end in .yml or .yaml.
    Returns:
        A dictionary containing the following MBIRJAX geometry parameters:
            - angles (ndarray): A 1D array of projection angles, in radians, specifying the angle of each projection relative to the origin.
            - sinogram_shape (tuple): Shape of the sinogram as a tuple in the form (views, rows, channels).
            - source_detector_dist (float): Distance between the X-ray source and the detector in units of :math:`ALU`.
            - source_iso_dist (float): Distance between the X-ray source and the center of rotation in units of :math:`ALU`.
            - magnification (float): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).
            - delta_det_channel (float): Detector channel spacing in :math:`ALU`.
            - delta_det_row (float): Detector row spacing in :math:`ALU`.
            - delta_voxel (float): Spacing between voxels in ALU.
            - det_channel_offset (float): Distance = (detector iso channel) - (center of detector channels) in ALU.
            - det_row_offset (float): Distance = (detector iso row) - (center of detector rows) in ALU.
    """

    ######### load yaml file into a dictionary
    if filename[-4:] == '.yml' or filename[-5:] == '.yaml':
        # Save the full parameter dictionary
        with open(filename, 'r') as file:
            yaml = YAML(typ="safe")
            params_dict_mbircone = yaml.load(file)
            params_dict_mbircone = convert_strings_to_arrays_mbircone(params_dict_mbircone)
    else:
        raise ValueError('Filename must end in .yaml or .yml: ' + filename)

    
    ######## convert MBIRCONE params to MBIRJAX params
    params_dict_mbirjax = param_convert_mbircone_to_mbirjax(params_dict_mbircone)
    
    print(f"from_file_mbircone(): MBIRJAX parameters:")
    pp.pprint(params_dict_mbirjax)

    return params_dict_mbirjax


def convert_strings_to_arrays_mbircone(cur_params):
    """
    Convert the string representation of an array back to an array. This function is modified from `mbirjax.ParameterHandler.convert_strings_to_arrays()`.
    Args:
        cur_params (dict): Parameter dictionary

    Returns:
        dict: The same dictionary with array strings replaced by arrays.
    """
    array_prefix = ParameterHandler.array_prefix
    for key, entry in cur_params.items():
        param_val = entry.get('val')
        # CHeck for a string with the array marker as prefix.
        if type(param_val) is str and param_val[0:len(array_prefix)] == array_prefix:
            # Strip the prefix, then remove the delimiters
            param_str = param_val[len(array_prefix):]
            clean_str = param_str.replace('[', '').replace(']', '').strip()
            # Read to a flat array, then reshape
            new_val = jnp.array(np.fromstring(clean_str + ' ', sep=' '))
            new_shape = cur_params[key]['shape']
            # Save the value and remove the 'shape' and 'val' keys, which are needed only for the yaml file.
            del cur_params[key]['shape']
            cur_params[key] = new_val.reshape(new_shape)
        else:
            cur_params[key] = param_val
            

    return cur_params


def param_convert_mbircone_to_mbirjax(params_dict_mbircone):
    """ This function converts MBIRCONE geometry parameters to MBIRJAX geometry parameters.
    Args:
        params_dict_mbircone: dictionary containing the MBIRCONE geometry parameters
    Returns: A dictionary containing the MBIRJAX geometry parameters.
        
    """
    params_dict_mbirjax = {}
    # directly copy parameters with same definitions in MBIRCONE
    params_dict_mbirjax["angles"] = params_dict_mbircone["angles"]
    params_dict_mbirjax["source_detector_dist"] = params_dict_mbircone["dist_source_detector"]
    params_dict_mbirjax["magnification"] = params_dict_mbircone["magnification"]
    params_dict_mbirjax["delta_det_channel"] = params_dict_mbircone["delta_det_channel"]
    params_dict_mbirjax["delta_det_row"] = params_dict_mbircone["delta_det_row"]
    params_dict_mbirjax["delta_voxel"] = params_dict_mbircone["delta_pixel_image"]
    params_dict_mbirjax["det_row_offset"] = params_dict_mbircone["det_row_offset"]
    
    # sinogram shape
    params_dict_mbirjax["sinogram_shape"] = (len(params_dict_mbircone["angles"]), params_dict_mbircone["num_det_rows"], params_dict_mbircone["num_det_channels"])
    
    # calculate det_channel_offset in mbirjax based on rotation_offset and det_channel_offset in mbircone
    params_dict_mbirjax["det_channel_offset"] = params_dict_mbircone["det_channel_offset"] + params_dict_mbircone["rotation_offset"]*params_dict_mbircone["magnification"]
    
    # calculate source_iso_dist based on magnification and dist_source_detector
    params_dict_mbirjax["source_iso_dist"] = params_dict_mbirjax["source_detector_dist"]/params_dict_mbirjax["magnification"]
    return copy.deepcopy(params_dict_mbirjax)
