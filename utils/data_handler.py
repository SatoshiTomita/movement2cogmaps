import os
import numpy as np
import cv2

def read_video_files_lq(video_dir, frame_dim, sigma=2, remove_bg=False):
    files = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if os.path.isfile(os.path.join(video_dir, f))]

    print(f'{video_dir}: ', end='')
    files = sorted(files)
    
    video_vector = np.zeros((len(files), 1, frame_dim[1], frame_dim[0]), dtype=np.float32)

    for idx, f in enumerate(files):
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if remove_bg:
            im = im[20:,:]
        im = cv2.GaussianBlur(im, ksize=(0,0), sigmaX=sigma)
        im = cv2.resize(im, frame_dim, interpolation=cv2.INTER_AREA)
        video_vector[idx, 0, ...] = (im / 255)
    
    print(f'{video_vector.shape}')
    
    return video_vector

def minmax_normalization(*data):
    for d in data:
        for idx in range(len(d)):
            if len(d[idx]) == 0 : continue
            d[idx] = (d[idx] - d[idx].min()) / (d[idx].max() - d[idx].min())
    return data

def create_multiple_subsampling(data, stride, is_velocity=False):
    new_length = data.shape[0]//stride if not is_velocity else data.shape[0]//stride-1
    data_multisubs = np.zeros(
        (stride, new_length, data.shape[1]),
        dtype=np.float32
    )
    for start_idx in range(stride):
        if is_velocity:
            if start_idx < stride-1:
                data_multisubs[start_idx] = data[start_idx+1:start_idx-stride+1].reshape(
                    new_length, stride, -1
                ).sum(axis=1)
            else:
                data_multisubs[start_idx] = data[start_idx+1:].reshape(
                    new_length, stride, -1
                ).sum(axis=1)
        else:
            data_multisubs[start_idx] = data[start_idx::stride]
    return data_multisubs