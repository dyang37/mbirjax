from ruamel.yaml import YAML
import numpy as np
import pprint
pp = pprint.PrettyPrinter(indent=4)

""" TODO: This function should be part of MBIRCONE package.
"""

def to_file_mbircone(filename,
                     angles, num_det_rows, num_det_channels, dist_source_detector, magnification,
                     delta_det_channel=1.0, delta_det_row=1.0, delta_pixel_image=None,
                     det_channel_offset=0.0, det_row_offset=0.0, rotation_offset=0.0):
    """ This function saves the MBIRCONE geometry parameters to a yaml file.
    Args:
        filename (str): Path to file to store the MBIRCONE parameter dictionary.  Must end in .yml or .yaml.
        angles (ndarray): A 1D array of projection angles, in radians, specifying the angle of each projection relative to the origin.
        num_det_rows (int): Number of detector rows.
        num_det_channels (int): Number of detector channels.
        dist_source_detector (float): Distance between the X-ray source and the detector in units of :math:`ALU`.
        magnification (float): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).
        
        delta_det_channel (float, optional): [Default=0.0] Detector channel spacing in :math:`ALU`.
        delta_det_row (float, optional): [Default=0.0] Detector row spacing in :math:`ALU`.
        delta_pixel_image (float, optional): [Default=None] Image pixel spacing in :math:`ALU`.
            If None, automatically set to ``delta_pixel_detector/magnification``.
        
        det_channel_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a row.
        det_row_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from center of detector to the source-detector line along a row.
        rotation_offset (float, optional): [Default=0.0] Distance in :math:`ALU` from source-detector line to axis of rotation in the object space.
    
    Returns:
        Nothing but creates or overwrites the specified file.
    """
    # construct a dictionary containing MBIRCONE geometry parameters
    params_dict_mbircone = {}
    params_dict_mbircone["angles"] = angles
    params_dict_mbircone["num_det_rows"] = num_det_rows
    params_dict_mbircone["num_det_channels"] = num_det_channels
    params_dict_mbircone["dist_source_detector"] = dist_source_detector
    params_dict_mbircone["magnification"] = magnification
    params_dict_mbircone["delta_det_channel"] = delta_det_channel
    params_dict_mbircone["delta_det_row"] = delta_det_row
    if delta_pixel_image is None:
        delta_pixel_image = delta_det_channel/magnification
    params_dict_mbircone["delta_pixel_image"] = delta_pixel_image
    params_dict_mbircone["det_channel_offset"] = det_channel_offset
    params_dict_mbircone["det_row_offset"] = det_row_offset
    params_dict_mbircone["rotation_offset"] = rotation_offset

    print(f"to_file_mbircone(): MBIRCONE parameters: ")
    pp.pprint(params_dict_mbircone)
    
    # save parameters as a yaml file
    output_params_mbircone = convert_arrays_to_strings_mbircone(params_dict_mbircone.copy()) # convert arrays to strings
    if filename[-4:] == '.yml' or filename[-5:] == '.yaml':
        # Save the full parameter dictionary
        with open(filename, 'w') as file:
            yaml = YAML()
            yaml.default_flow_style = False
            yaml.dump(output_params_mbircone, file)
    else:
        raise ValueError('Filename must end in .yaml or .yml: ' + filename)

def convert_arrays_to_strings_mbircone(cur_params):
    """
    Replaces any jax or numpy arrays in cur_params with a flattened string representation and the array shape. This function is modified from `mbirjax.parameter_handler.convert_arrays_to_strings()`.
    Args:
        cur_params (dict): Parameter dictionary

    Returns:
        dict: The same dictionary with arrays replaced by strings.
    """
    array_prefix = ':ARRAY:'
    for key, param_val in cur_params.items():
        if type(param_val) == type(np.ones(1)):
            # Get the array values, then flatten them and put them in a string.
            cur_array = np.array(param_val)
            formatted_string = " ".join(f"{x:.7f}" for x in cur_array.flatten())
            # Include a prefix for identification upon reading
            new_val = array_prefix + formatted_string
            cur_params[key] = {'val': new_val, 'shape': param_val.shape}
        else:
            cur_params[key] = {'val': param_val}
    return cur_params
