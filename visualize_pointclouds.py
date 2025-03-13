import rerun as rr
import time
import numpy as np
from A2_starter_code import *

def read_xyz(filenm, path = 'pointclouds-500'):
    """
    Reading points
        filenm: the file name
    """
    filenm = path + '\\' + filenm

    points = []
    with open(filenm, 'r') as f_input:
        for line in f_input:
            p = line.split()
            p = [float(i) for i in p]
            points.append(p)
    points = np.array(points).astype(np.float32)
    return points

def choose_colors(based_on, object, min_max = []):
    if based_on == 'random':
        colors = [
            np.random.randint(50, 255),
            np.random.randint(50, 255),
            np.random.randint(50, 255),
        ]

    if based_on == 'object_class':
        label = object.label
        if label == 0:
            colors = [178, 34, 34]
        if label == 1:
            colors = [128,128,128]
        if label == 2:
            colors = [160, 34, 240]
        if label == 3:
            colors = [30,144,255]
        if label == 4:
            colors = [107, 142, 35]

    if based_on == 'feature':
        min, max, feature = min_max
        x = object.feature[feature]
        x_normalized = 128 + ((x - min)*(255-128))/(max-min)
        colors = [x_normalized,x_normalized,x_normalized]

    return colors


def visualize(objects, based_on = 'random', sleeptime=0.01, show = [0,1,2,3,4]):
    # -- init rerun viewer
    rr.init("Regiongrowing Results", spawn=True)

    if based_on == 'feature':
        val = int(input("What feature do you want to color with: "))
        feature_list = []
        for object in objects:
            feature_list.append(object.feature[val])

        min_max = [min(feature_list), max(feature_list), val]

    # -- log pointcloud one-by-one
    for object in objects:
        if object.label not in show:
            continue

        if based_on == 'feature':
            colors = choose_colors(based_on, object, min_max)
        else:
            colors = choose_colors(based_on, object)

        subset = object.points[:, :3]
        rr.log(
            "segment_{}".format(object.cloud_ID),
            rr.Points3D(
                subset[:],
                colors=colors,
                radii=0.1,
            ),
        )
        rr.log(
            "logs_{}".format(object.cloud_ID),
            rr.TextLog(
                "size segment_{}=={}".format(object.cloud_ID, subset.shape[0]),
                level=rr.TextLogLevel.TRACE,
            ),
        )
        time.sleep(sleeptime)

def create_urban_objects(data_path):
    object_list = []
    # obtain the files in the folder
    files = sorted(listdir(data_path))

    # initialize the data
    input_data = []

    # retrieve each data object and obtain the feature vector
    for file_i in tqdm(files, total=len(files)):
        # obtain the file name
        file_name = join(data_path, file_i)

        # read data
        i_object = urban_object(filenm=file_name)

        # calculate features
        i_object.compute_features()

        # add the data to the list
        object_list.append(i_object)

    return object_list

object_list = create_urban_objects('pointclouds-500')

#for object in object_list:
#    print(object.cloud_ID)

visualize(object_list, based_on='random')