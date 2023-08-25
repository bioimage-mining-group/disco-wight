import numpy as np
import glob
import os
import shutil
import scipy.ndimage as nd
from tqdm import tqdm
import tifffile


def load_csv_coords(path_to_csv: str):
    """Loads coordinates from a CSV file.

    Args:
        path_to_csv (str): Path to the CSV file.
    """
    return np.loadtxt(fname=path_to_csv, delimiter=',')


def delete_content_of_dir(path_to_dir: str, content_type: str = ''):
    """
    Deletes content of a directory.

    Args:
        path_to_dir (str): the path to the directory whose content is to be deleted.
        content_type (str, optional): can be given to filter deleted content by end of filename. Defaults to '', in which case all content is deleted.
    """
    files = glob.glob(f'{path_to_dir}/*{content_type}')
    
    print(f'Deleting content of folder {path_to_dir}')
    if len(files)>0:
    
        for f in files:
            try:
                os.remove(f)
            except IsADirectoryError:
                shutil.rmtree(f)

def make_bounding_box(bb_shape: tuple, bb_width: int = 1):
    """
    Makes an array of shape 'bb_shape' with 1s at the edges
    with width 'bb_width' and 0s everywhere else.
    
    """

    if len(bb_shape) == 2:

        bounding_box = np.ones(bb_shape, dtype='uint8')
        
        bounding_box[bb_width:-bb_width, bb_width:-bb_width] = 0

    elif len(bb_shape) == 3:
        
        bounding_box = np.ones(bb_shape, dtype='uint8')
        
        # actually don't put 'bb_width' into this, so
        # that it produces a nice looking "cornered"
        # bounding box
        bounding_box[1:-1, 1:-1, 1:-1] = 0
        
        bounding_box[[0, -1], bb_width : -bb_width, bb_width : -bb_width] = 0
        bounding_box[bb_width : -bb_width, [0, -1], bb_width : -bb_width] = 0
        bounding_box[bb_width : -bb_width, bb_width : -bb_width, [0, -1]] = 0
        
    elif len(bb_shape) == 4:

        bounding_box = make_bounding_box(bb_shape[1:], bb_width)
        
        bounding_box = repeat_along_t(bounding_box, repeat=bb_shape[0])
        
    else:
        print(f'Inputs with dim {len(bb_shape)} are not implemented.')
        raise NotImplementedError
    
    return bounding_box

def repeat_along_t(array, repeat: int):
    """
    Duplicates input array along first dimension.
    """
    return np.stack((array,) * repeat, axis=0)


def rescale_array(input_array, zoom_factor: tuple, order: int):
    """
    Rescale 'input_array' by zoom_factor, using interpolation of
    given 'order'.
    """

    if order == 0:
        input_array = input_array.astype(int)
    else:
        input_array = input_array.astype(float)

    if input_array.ndim >3:

        resampled = np.array(
            [nd.zoom(stack, zoom_factor, order=order) for stack in input_array]
        )

    else:
        resampled = nd.zoom(input_array, zoom_factor, order=order)

    if order == 0:
        resampled  = resampled.astype("uint32")

    return resampled

def array_to_utrack3d_inputs(array, path_to_save: str, transpose: bool = False, trailing_zeros: bool = True):
    """
    Saves each frames of a given array of format T[rest of the coords]
    at given path with standardized filenames.
    """

    delete_content_of_dir(path_to_dir=path_to_save, content_type='.tif')

    num_t = len(array)

    for tind, stack in enumerate(tqdm(array, desc='Saving frames')):

        if trailing_zeros:
            standardized_id = str(tind).zfill(int(np.log10(num_t)) + 1)
        else:
            standardized_id = str(tind)

        if transpose:
            stack = np.transpose(stack,axes=(2,1,0))

        tifffile.imwrite(
            f'{path_to_save}/label-frame-{standardized_id}.tif',
            stack
        )
