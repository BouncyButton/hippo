import nibabel as nib
# https://www.nitrc.org/projects/hippseg_2011/
from ipywidgets import interact, IntSlider
import numpy as np

img = nib.load("HFH_001_Hipp_Labels.img")  # .hdr must be in the same directory
data = img.get_fdata()
affine = img.affine

# show a slice
from matplotlib import pyplot as plt

plt.imshow(data[:, :, data.shape[2] // 2])
plt.show()

import pyvista as pv

spacing = np.abs(affine[:3, :3].diagonal())
origin = affine[:3, 3]

data = data[:, :, :, 0]

grid = pv.ImageData()
grid.dimensions = data.shape  # point dimensions
grid.spacing = spacing
grid.origin = origin

grid.point_data["labels"] = data.flatten(order="F")

plotter = pv.Plotter()
plotter.add_volume(grid, opacity="sigmoid", shade=True)
plotter.show()
