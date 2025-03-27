# GEO5017 assignment 2

## Pointcloud file format: 

Each 'xyz' file contains the point cloud of a single object, in which each line has three floating point numbers denoting the x, y, and z coordinates of a 3D point.
Ground truth labels:
```
000 - 099: building
100 - 199: car
200 - 299: fence
300 - 399: pole
400 - 499: tree
```

## Needed external libraries:
```
import rerun as rr
import time
import numpy as np
import sklearn
import tqdm
import matplotlib.pyplot as plt
```

## How to use the provided code
Two different files are included in the Github repository. The A2_starter_cody.py is the main file to run the machine learning code, while visualize_pointclouds.py can be used to visualize the entire provided dataset. 

### Running A2_starter_cody.py
By inputting a key number (0 - 16) corresponding to the 16 identified features in our paper, the program visualizes the distribution of the inputted feature for each of the 5 different classes. 

By pressing -1, the program continues by conducting feature selection, followed by hyperparameter tuning and then the visualizing of the corresponding learning curve. It does this for both the SVM and the RF models.

We recommend running the program multiple times to see how the results can differ and change based on the random selection of training data and of course the tuning of hyperparameters. 

### Running visualize_pointclouds.py
When running this .py file, the function automatically visualizes the entire point cloud with random colours. By changing the based_on input value in the visualize() function, you can change the colouring of the objects. By changing the value to 'feature' each pointcloud will be visualized on a gray scale based on how much the normalized feature value is. To choose a feature, input a key number (0 - 16) corresponding to the 16 identified features in our paper. Lastly, it is also possible to input 'object_class', to colour each point cloud in the same class the same colour. 

## Credit and references:

All point clouds are taken from the DALES Objects dataset. More details about the dataset can be found in the following paper:
Singer et al. DALES Objects: A Large Scale Benchmark Dataset for Instance Segmentation in Aerial Lidar, 2021.
