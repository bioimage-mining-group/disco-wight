<img src="imgs/dwight2.png" width="150">  

# disco-wight (or **Dwight** for short)
## is a **D**etector-**I**ndependent quality **SCO**re **WI**thout **G**round **T**rut**H**

This repository contains the scripts used in the paper  
*A detector-independent quality score for cell segmentation without ground truth in 3D live fluorescence microscopy*, Vanaret et al. (2023)

# /!\ **This repository will be updated with the scripts in the following months**  

Major dependancies to be installed are :
- [numpy] : basic library for array manipulation
- [matplotlib] : basic library to plot figures
- [tifffile] : library to read and write tiff images
- [napari] : 3D image visualizer
- [scipy] : general library for scientific python
- [numba] : library to make python fast

From your main Conda environment (or from a custom one), refer to the dependancies' individual documentations to install them. This can usually be done easily using [pip].

To install the repositiory package:

```bash
# from repo root (the DOT "." is important !!!)
cd dwight;pip install -e .
```



[pip]: https://pypi.org/project/pip
[numpy]: https://numpy.org
[matplotlib]: https://matplotlib.org
[tifffile]: https://pypi.org/project/tifffile
[napari]: https://napari.org
[scipy]: https://scipy.org/
[numba]: https://numba.pydata.org/
