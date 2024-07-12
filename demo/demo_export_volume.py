import numpy as np
import os
import tifffile

def export_volume(array, file_name, file_type, slice_axis=2):
    """
    Export a 3D volume to a file.

    Parameters
    ----------
    array : numpy.ndarray
        The 3D volume to be exported.
    file_name : str
        The name of the output file.
    file_type : str
        The type of file to export. Must be one of 'npy', 'tiff', or 'tiff_stack'.
        
        - 'npy': Saves the entire 3D array as a single .npy file.
        - 'tiff': Saves the entire 3D array as a single multi-page TIFF file.
        - 'tiff_stack': Saves each slice of the 3D array (along the specified axis) 
          as an individual TIFF file in a directory named after the base of `file_name`.

    slice_axis : int, optional
        The axis along which to slice the array for `tiff_stack`. Default is 2.

    Raises
    ------
    ValueError
        If the file_type is not one of 'npy', 'tiff', or 'tiff_stack'.
        If the slice_axis is out of bounds for the array dimensions.

    Examples
    --------
    >>> array = np.random.rand(100, 100, 10)
    >>> export_volume(array, 'output.npy', 'npy')
    >>> export_volume(array, 'output.tiff', 'tiff')
    >>> export_volume(array, 'output.tiff', 'tiff_stack', slice_axis=2)
    """
    if file_type not in ['npy', 'tiff', 'tiff_stack']:
        raise ValueError("file_type must be one of 'npy', 'tiff', or 'tiff_stack'")
    
    # Ensure the slice_axis is valid
    if slice_axis < 0 or slice_axis >= array.ndim:
        raise ValueError("slice_axis is out of bounds for the array dimensions")

    # Save as npy
    if file_type == 'npy':
        np.save(file_name, array)
        print(f"Data exported to {file_name}")
    
    # Save as a single tiff file
    elif file_type == 'tiff':
        tifffile.imwrite(file_name, array)
        print(f"Data exported to {file_name}")

    # Save as a tiff stack
    elif file_type == 'tiff_stack':
        # Extract directory and base name from file_name
        base_name = os.path.splitext(os.path.basename(file_name))[0]
        base_dir = os.path.dirname(file_name)
        output_dir = os.path.join(base_dir, base_name)
        
        # Create the output directory
        os.makedirs(output_dir, exist_ok=True)

        # Permute the array to have slice_axis as the last dimension
        permuted_array = np.moveaxis(array, slice_axis, -1)

        # Save each slice as a separate tiff file
        print(f"Exporting data as a tiff stack ...")
        for i in range(permuted_array.shape[-1]):
            slice_file_name = os.path.join(output_dir, f"{base_name}{i+1:03d}.tiff")
            tifffile.imwrite(slice_file_name, permuted_array[..., i])
            print(f"Exported {slice_file_name}")



# Example usage
if __name__ == "__main__":
    array = np.random.rand(100, 100, 10)  # Example 3D array
    os.makedirs("output", exist_ok=True)
    export_volume(array, 'output/saved_data.npy', 'npy')
    export_volume(array, 'output/saved_data.tiff', 'tiff')
    export_volume(array, 'output/saved_data.tiff', 'tiff_stack', slice_axis=2)
