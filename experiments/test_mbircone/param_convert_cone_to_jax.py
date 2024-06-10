import pickle
import pprint
pp = pprint.PrettyPrinter(indent=4)

def write_mbircone_params(filename,
                          angles, num_det_rows, num_det_channels, dist_source_detector, magnification,
                          delta_det_channel=1.0, delta_det_row=1.0, delta_pixel_image=None,
                          det_channel_offset=0.0, det_row_offset=0.0, rotation_offset=0.0):
    """ This function writes MBIRCONE geometry parameters into a pickle file.
    Args:
        filename (string): A path-like object (https://docs.python.org/3/glossary.html#term-path-like-object) giving the path to the pickle file which contains the MBIRCONE geometry parameters.
        
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
    
    Returns: None
    """
    # construct a dictionary containing MBIRCONE geometry parameters
    geo_params_mbircone = {}
    geo_params_mbircone["angles"] = angles
    geo_params_mbircone["num_det_rows"] = num_det_rows
    geo_params_mbircone["num_det_channels"] = num_det_channels
    geo_params_mbircone["dist_source_detector"] = dist_source_detector
    geo_params_mbircone["magnification"] = magnification
    geo_params_mbircone["delta_det_channel"] = delta_det_channel
    geo_params_mbircone["delta_det_row"] = delta_det_row
    if delta_pixel_image is None:
        delta_pixel_image = delta_det_channel/magnification
    geo_params_mbircone["delta_pixel_image"] = delta_pixel_image
    geo_params_mbircone["det_channel_offset"] = det_channel_offset
    geo_params_mbircone["det_row_offset"] = det_row_offset
    geo_params_mbircone["rotation_offset"] = rotation_offset

    print(f"write_mbircone_params(): the following parameters are saved into {filename}: ")
    pp.pprint(geo_params_mbircone)
    # save the dictionary as a pickle file
    with open(filename, 'wb') as f:
        pickle.dump(geo_params_mbircone, f)
    return

def read_mbircone_params(filename):
    """ This function read MBIRCONE geometry parameters from a pickle file.
    Args:
        filename (string): A path-like object (https://docs.python.org/3/glossary.html#term-path-like-object) giving the path to the pickle file which contains the MBIRCONE geometry parameters.
    Returns:
        geo_params_mbircone: A dictionary containing the following MBIRCONE geometry parameters:
            - angles (ndarray): A 1D array of projection angles, in radians, specifying the angle of each projection relative to the origin.
            - num_det_channels (int): Number of detector channels.
            - num_det_rows (int): Number of detector rows.
            - dist_source_detector (float): Distance between the X-ray source and the detector in units of :math:`ALU`.
            - magnification (float): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).
            - delta_det_channel (float): Detector channel spacing in :math:`ALU`.
            - delta_det_row (float): Detector row spacing in :math:`ALU`.
            - delta_pixel_image (float): Image pixel spacing in :math:`ALU`.
            - det_channel_offset (float): Distance in :math:`ALU` from center of detector to the source-detector line along a row.
            - det_row_offset (float): Distance in :math:`ALU` from center of detector to the source-detector line along a row.
            - rotation_offset (float): Distance in :math:`ALU` from source-detector line to axis of rotation in the object space.
            - image_slice_offset (float): Vertical offset of the image in units of :math:`ALU`. 
    """

    with open('saved_dictionary.pkl', 'rb') as f:
        geo_params_mbircone = pickle.load(f)
    
    print(f"read_mbircone_params(): the following MBIRCONE parameters are loaded from {filename}: ")
    pp.pprint(geo_params_mbircone)

    return geo_params_mbircone

def param_convert_cone_to_jax(geo_params_mbircone):
    """ This function converts MBIRCONE parameters to MBIRJAX parameters.

    Args:
        geo_params_mbircone: A dictionary containing the following MBIRCONE geometry parameters:
            - angles (ndarray): A 1D array of projection angles, in radians, specifying the angle of each projection relative to the origin.
            - num_det_channels (int): Number of detector channels.
            - num_det_rows (int): Number of detector rows.
            - dist_source_detector (float): Distance between the X-ray source and the detector in units of :math:`ALU`.
            - magnification (float): Magnification of the cone-beam geometry defined as (source to detector distance)/(source to center-of-rotation distance).
            - delta_det_channel (float): Detector channel spacing in :math:`ALU`.
            - delta_det_row (float): Detector row spacing in :math:`ALU`.
            - delta_pixel_image (float): Image pixel spacing in :math:`ALU`.
            - det_channel_offset (float): Distance in :math:`ALU` from center of detector to the source-detector line along a row.
            - det_row_offset (float): Distance in :math:`ALU` from center of detector to the source-detector line along a row.
            - rotation_offset (float): Distance in :math:`ALU` from source-detector line to axis of rotation in the object space.
            - image_slice_offset (float): Vertical offset of the image in units of :math:`ALU`. 
   
    Returns:
        geo_params_mbirjax: A dictionary containing the following MBIRJAX geometry parameters:
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
    geo_params_mbirjax = {} # dictionary that contains the MBIRJAX geometry parameters
    
    # parameters with the same definitions can be copied directly
    geo_params_mbirjax["angles"] = geo_params_mbircone["angles"]
    geo_params_mbirjax["source_detector_dist"] = geo_params_mbircone["dist_source_detector"]
    geo_params_mbirjax["magnification"] = geo_params_mbircone["magnification"]
    geo_params_mbirjax["delta_det_channel"] = geo_params_mbircone["delta_det_channel"]
    geo_params_mbirjax["delta_det_row"] = geo_params_mbircone["delta_det_row"]
    geo_params_mbirjax["delta_voxel"] = geo_params_mbircone["delta_pixel_image"]
    geo_params_mbirjax["det_row_offset"] = geo_params_mbircone["det_row_offset"]

    # get sinogram_shape from MBIRCONE parameters
    num_views = len(angles)
    num_det_rows = geo_params_mbircone["num_det_rows"]
    num_det_channels = geo_params_mbircone["num_det_channels"]
    geo_params_mbirjax["sinogram_shape"] = (num_views, num_det_rows, num_det_channels)
    
    # calculate det_channel_offset in mbirjax based on rotation_offset and det_channel_offset in mbircone
    geo_params_mbirjax["det_channel_offset"] = geo_params_mbircone["det_channel_offset"] + geo_params_mbircone["rotation_offset"]*geo_params_mbircone["magnification"]
    
    # calculate source_iso_dist based on magnification and dist_source_detector
    geo_params_mbirjax["source_iso_dist"] = geo_params_mbirjax["source_detector_dist"]/geo_params_mbirjax["magnification"]
     
    print(f"param_convert_cone_to_jax(): MBIRJAX parameters:")
    pp.pprint(geo_params_mbirjax)

    return geo_params_mbirjax
