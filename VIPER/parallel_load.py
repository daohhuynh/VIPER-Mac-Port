import zarr
import numpy as np
import os
import sys
from scipy import signal
try:
    import cupy as cp
    from cupyx.scipy import signal as csignal
except:
    import numpy as cp
    import scipy.signal as csignal
    pass
import tifffile as tif
import sys


def downsample2d(inputArray, kernelSize, gpu_mode=False):

    """
    Downsample 2D IMage
    
    Parameters
    ----------
    inputArray : Input Image
    kernelSize : Downsampling factor

    Returns
    -------
    downsampled_array
    """

    n,m = inputArray.shape
    data_type = inputArray.dtype
    row_crop = n%kernelSize
    col_crop = m%kernelSize
    if gpu_mode:
        average_kernel = (1/(kernelSize*kernelSize))*cp.ones((kernelSize,kernelSize),dtype=data_type)
        blurred_array = csignal.convolve2d(inputArray, average_kernel, mode='same')
    else:
        average_kernel = (1/(kernelSize*kernelSize))*np.ones((kernelSize,kernelSize),dtype=data_type)
        blurred_array = signal.convolve2d(inputArray, average_kernel, mode='same')

    
    if row_crop > 0:
        blurred_array = blurred_array[row_crop:,:]
    else:
        pass
    if col_crop > 0:
        blurred_array = blurred_array[:,col_crop:]
    else:
        pass
    
    downsampled_array = blurred_array[::kernelSize,::kernelSize]

    return downsampled_array

def downsample3d(inputArray, kernelSize, gpu_mode=False):
    """
    Downsample Video (3D Array)
    
    Parameters
    ----------
    inputArray : Input Array (3D)
    kernelSize : Downsampling Factor (for each 2D image)

    Returns
    -------
    downsampled array

    """
    first_downsampled = downsample2d(inputArray[0,:,:], kernelSize, gpu_mode=gpu_mode)
    data_type = first_downsampled.dtype
    if gpu_mode:
        downsampled_array = cp.zeros((inputArray.shape[0],first_downsampled.shape[0], first_downsampled.shape[1]),dtpye = data_type)
    else:
        downsampled_array = np.zeros((inputArray.shape[0],first_downsampled.shape[0], first_downsampled.shape[1]),dtpye = data_type)

    downsampled_array[0,:,:] = first_downsampled

    for i in range(1, inputArray.shape[0]):
        downsampled_array[i,:,:] = downsample2d(inputArray[i,:,:], kernelSize, gpu_mode=gpu_mode)

    return downsampled_array

def map_frames(imageData, start_idx, files, ds, gpu_mode):
    
    for i, file in enumerate(files):
        image = tif.imread(file)
        if ds > 1:
            if gpu_mode:
                image = downsample2d(cp.array(image), ds, gpu_mode=gpu_mode)
                imageData[start_idx+i, :, :] = cp.asnumpy(image)
            else:
                image = downsample2d(image, ds, gpu_mode=gpu_mode)
                imageData[start_idx+i, :, :] = image
        else:
            imageData[start_idx+i, :, :] = image

def map_chunks(imageData, start_idx, chunk, ds, gpu_mode):

    for i in range(len(chunk)):
        image = chunk[i]
        if ds > 1:
            if gpu_mode:
                image = downsample2d(cp.array(image), ds, gpu_mode=gpu_mode)
                imageData[start_idx+i, :, :] = cp.asnumpy(image)
            else:
                image = downsample2d(image, ds, gpu_mode=gpu_mode)
                imageData[start_idx+i, :, :] = image
        else:
            imageData[start_idx+i, :, :] = image


if __name__ == "__main__":

    mappedImageDataPath = sys.argv[1]
    start = int(sys.argv[2])
    filePath = sys.argv[3]
    ds = int(sys.argv[4])
    gpu_mode = bool(int(sys.argv[5]))
    data_shape = sys.argv[6]

    imageData = zarr.open_array(mappedImageDataPath, mode='a')

    if data_shape == 'frame':
        frames = eval(filePath)
        map_frames(imageData, start, frames, ds, gpu_mode)
    elif data_shape == 'chunk':
        chunk = tif.imread(filePath, aszarr=True)
        chunk = zarr.open_array(chunk, mode='r')
        map_chunks(imageData, start, chunk, ds, gpu_mode)



