"""
This demo shows how to visualize the designed features. Currently, only 2D feature space visualization is supported.
I use the same data for A2 as my input.
Each .xyz file is initialized as one urban object, from where a feature vector is computed.
6 features are defined to describe an urban object.
Required libraries: numpy, scipy, scikit learn, matplotlib, tqdm 
"""

import math
from os import listdir
from os.path import exists, join

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import ConvexHull
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KDTree
from tqdm import tqdm


class urban_object:
    """
    Define an urban object
    """
    def __init__(self, filenm):
        """
        Initialize the object
        """
        # obtain the cloud name
        self.cloud_name = filenm.split('/\\')[-1][-7:-4]

        # obtain the cloud ID
        self.cloud_ID = int(self.cloud_name)

        # obtain the label
        self.label = math.floor(1.0*self.cloud_ID/100)

        # obtain the points
        self.points = read_xyz(filenm)

        # initialize the feature vector
        self.feature = []

    def compute_features(self):
        # calculate the height
        height = np.amax(self.points[:, 2])
        self.feature.append(height)

        # get the root point and top point
        root = self.points[[np.argmin(self.points[:, 2])]]
        # top = self.points[[np.argmax(self.points[:, 2])]]

        # construct the 2D and 3D kd tree
        kd_tree_2d = KDTree(self.points[:, :2], leaf_size=5)
        # kd_tree_3d = KDTree(self.points, leaf_size=5)

        # compute the root point planar density
        radius_root = 0.2
        count = kd_tree_2d.query_radius(root[:, :2], r=radius_root, count_only=True)
        root_density = 1.0*count[0] / len(self.points)
        self.feature.append(root_density)

        # compute the 2D footprint and calculate its area
        hull_2d = ConvexHull(self.points[:, :2])
        hull_area_2d = hull_2d.volume
        self.feature.append(hull_area_2d)

        # get the hull shape index
        hull_perimeter = hull_2d.area
        shape_index = 1.0 * hull_area_2d / hull_perimeter
        self.feature.append(shape_index)

        # obtain the covariance matrix and eigenvalues
        cov = np.cov(self.points.T)
        w, _ = np.linalg.eig(cov)
        w.sort()

        # calculate the metrics described in "Contour detection in unstructured 3D point clouds"
        # 0-3, 2-1, 1-2
        sum_of_eigenvalues = np.sum(w)
        linearity = (w[2]-w[1]) / (w[2] + 1e-5)
        planarity = (w[1] - w[0]) / (w[2] + 1e-5)
        sphericity = w[0] / (w[2] + 1e-5)
        omnivariance = pow((w[2] * w[1] * w[0]), float(1.0 / 3.0))
        anisotropy = (w[2] - w[0]) / (w[2] + 1e-5)
        change_of_curvature = (w[0] / sum_of_eigenvalues)
        eigenetropy = 0
        for eigenvalue in w:
            eigenetropy += eigenvalue * math.log(eigenvalue)
        eigenetropy *= -1

        self.feature += [sum_of_eigenvalues, linearity, planarity, sphericity, omnivariance, anisotropy, change_of_curvature, eigenetropy]

        # add the number of points in the point cloud
        self.feature.append(len(self.points))

        # estimate 3d volume
        hull_3d = ConvexHull(self.points)
        hull_volume_3d = hull_3d.volume
        self.feature.append(hull_volume_3d)

        # calculate average height
        avg_height = np.average(self.points[:, 2])
        self.feature.append(avg_height)

        # calculate height, x, and y standard deviation
        std_x = np.std(self.points[:, 0])
        std_y = np.std(self.points[:, 1])
        std_height = np.std(self.points[:, 2])
        self.feature += [std_x + std_y, std_height]

def read_xyz(filenm):
    """
    Reading points
        filenm: the file name
    """
    points = []
    with open(filenm, 'r') as f_input:
        for line in f_input:
            p = line.split()
            p = [float(i) for i in p]
            points.append(p)
    points = np.array(points).astype(np.float32)
    return points


def feature_preparation(data_path):
    """
    Prepare features of the input point cloud objects
        data_path: the path to read data
    """
    # check if the current data file exist
    data_file = 'data.txt'
    if exists(data_file):
        return

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
        i_data = [i_object.cloud_ID, i_object.label] + i_object.feature
        input_data += [i_data]

    # transform the output data
    outputs = np.array(input_data).astype(np.float32)

    # write the output to a local file
    data_header = ('ID,label,height,root_density,hull_area_2d,shape_index,sum_of_eigenvalues,linearity,planarity,'
                   'sphericity,omnivariance,anisotropy,change_of_curvature,eigenetropy,num_points,'
                   'hull_volume_3d, avg_height, std_x_y, std_height')
    np.savetxt(data_file, outputs, fmt='%10.5f', delimiter=',', newline='\n', header=data_header)


def data_loading(data_file='data.txt'):
    """
    Read the data with features from the data file
        data_file: the local file to read data with features and labels
    """
    # load data
    data = np.loadtxt(data_file, dtype=np.float32, delimiter=',', comments='#')

    # extract object ID, feature X and label Y
    ID = data[:, 0].astype(np.int32)
    y = data[:, 1].astype(np.int32)
    X = data[:, 2:].astype(np.float32)

    return ID, X, y

def feature_selection(X):
    features = np.copy(X).T
    feature_set = []
    N_N_k = float(1/5)

    # Perform forward search for feature selection
    while len(feature_set) < 4:
        feature_J_val = []

        for feature in features:
            total_mean = np.mean(feature)
            sw = 0
            sb = 0
            # Calculate within and between class scatter matrix
            for i in range(5):
                class_features = feature[100 * i:100 * (i + 1)]
                if len(feature_set) == 0:
                    # Todo: is this correct? How to calculate values when just one feature is being analysed
                    # Within class
                    # (np.cov(np.array([class_features, class_features]).T)???) Just 100x100 0's, divide by 0 error
                    cov = np.cov([class_features, class_features])
                    sw += (N_N_k * cov)
                    # Between class
                    # Just the sample values?
                    sample_means = np.array([class_features])
                    sb += (N_N_k * ((sample_means * total_mean) * (sample_means * total_mean).T))
                else:
                    # Within class
                    current_features = np.array(feature_set)[:,100 * i:100 * (i + 1)]
                    # Todo: Transpose?? 100x100 or 2x2
                    cov = np.cov(np.append(current_features, [class_features], axis=0))
                    sw += (N_N_k * cov)
                    # Between class
                    # Todo: correct axis??
                    sample_means = np.array([np.mean(np.append(current_features, [class_features], axis=0), axis=0)])
                    sb += (N_N_k * ((sample_means * total_mean) * (sample_means * total_mean).T))

            feature_J_val.append(np.trace(sb) / np.trace(sw))

        # Sort features based on best J value
        ind = np.argsort(feature_J_val)
        features = features[ind]
        # Select best feature and add to feature set
        best_feature = features[-1]
        features = np.delete(features, len(features) - 1, axis=0)
        feature_set.append(best_feature)

    # Return feature set
    feature_set = np.array(feature_set).T
    return feature_set


def feature_visualization(X):
    """
    Visualize the features
        X: input features. This assumes classes are stored in a sequential manner
    """
    # define the labels and corresponding colors
    colors = ['firebrick', 'grey', 'darkorange', 'dodgerblue', 'olivedrab']
    labels = ['building', 'car', 'fence', 'pole', 'tree']
    features = ['height','root_density','hull_area_2d','shape_index','sum_of_eigenvalues','linearity','planarity',
                   'sphericity','omnivariance','anisotropy','change_of_curvature','eigenetropy','num_points',
                   'hull_volume_3d', 'avg_height', 'std_x_y', 'std_height']

    while True:
        print(f"\nPlease select a feature to visualize by pressing the corresponding number key (0 - {len(X[0]) - 1})")
        print("Otherwise, press -1 to continue")
        try:
            choice = int(input("Enter your choice: "))
            if choice < -1 or choice > len(X[0]) - 1:
                print("Invalid choice. Please select a valid option.\n")
                continue
            if choice == -1:
                print("Exiting visualization.")
                break
            else:
                # initialize a plot
                fig = plt.figure()
                ax = fig.add_subplot()
                plt.title("feature subset visualization of 5 classes", fontsize="small")

                for i in range(5):
                    ax.scatter(X[100 * i:100 * (i + 1), choice], (X[100 * i:100 * (i + 1), 0] * 0 + i), marker="o",
                               c=colors[i],
                               edgecolor="k", label=labels[i])

                ax.set_ylabel('x1:label')
                ax.set_xlabel(f'x2:feature {features[choice]}')
                ax.legend()
                plt.show()

        except ValueError:
            print("Invalid input. Please enter a number corresponding to an option.\n")


def SVM_classification(X, y):
    """
    Conduct SVM classification
        X: features
        y: labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clf = svm.SVC()
    clf.fit(X_train, y_train)
    y_preds = clf.predict(X_test)
    acc = accuracy_score(y_test, y_preds)
    print("SVM accuracy: %5.2f" % acc)
    print("confusion matrix")
    conf = confusion_matrix(y_test, y_preds)
    print(conf)


def RF_classification(X, y):
    """
    Conduct RF classification
        X: features
        y: labels
    """
    pass


if __name__=='__main__':
    # specify the data folder
    """"Here you need to specify your own path"""
    path = 'pointclouds-500'

    # conduct feature preparation
    print('Start preparing features')
    feature_preparation(data_path=path)

    # load the data
    print('Start loading data from the local file')
    ID, X, y = data_loading()

    # visualize features
    print('Visualize the features')
    feature_visualization(X=X)

    # conduct feature selection
    print('Start selection features')
    refined_X = feature_selection(X)

    # SVM classification
    print('Start SVM classification')
    SVM_classification(refined_X, y)

    # RF classification
    print('Start RF classification')
    RF_classification(refined_X, y)