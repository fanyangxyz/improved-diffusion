import sys
import os

import numpy as np
from matplotlib import pyplot as plt

file_path = sys.argv[1]
print(file_path)

arr = None
with np.load(file_path) as data:
    for item in data:
        print(item)
        print(data[item].shape)
        arr = data[item]
        break

assert arr is not None
file_name = file_path.split('/')[-1].split('.')[0]
file_folder = file_path.rsplit('/', 1)[0]
print(file_name)
print(file_folder)
image_dir = os.path.join(file_folder, 'images')
os.makedirs(image_dir, exist_ok=True)
print(image_dir)

def save_image(sample, nrow, name):
    assert sample.dtype == 'float32', sample.dtype
    sample = np.clip(sample, 0, 1)
    ncol = len(sample) // nrow
    fig, axes = plt.subplots(nrow, ncol)
    for i, ax in enumerate(axes.flat):
        ax.imshow(sample[i])
        ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    fig.savefig(name)
    plt.close('all')

batch_size = len(arr)
fig, ax = plt.subplots(batch_size, 1)
#for i, ax in enumerate(axes.flat):
ax.imshow(arr[0])
ax.axis('off')

plt.subplots_adjust(wspace=0, hspace=0)
fig.savefig(os.path.join(image_dir, 'sampled_images.png'))
