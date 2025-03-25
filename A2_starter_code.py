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
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
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
        self.label = math.floor(1.0 * self.cloud_ID / 100)

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
        root_density = 1.0 * count[0] / len(self.points)
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
        linearity = (w[2] - w[1]) / (w[2] + 1e-5)
        planarity = (w[1] - w[0]) / (w[2] + 1e-5)
        sphericity = w[0] / (w[2] + 1e-5)
        omnivariance = pow((w[2] * w[1] * w[0]), float(1.0 / 3.0))
        anisotropy = (w[2] - w[0]) / (w[2] + 1e-5)
        change_of_curvature = (w[0] / sum_of_eigenvalues)
        eigenetropy = 0
        for eigenvalue in w:
            eigenetropy += eigenvalue * math.log(eigenvalue)
        eigenetropy *= -1

        self.feature += [sum_of_eigenvalues, linearity, planarity, sphericity, omnivariance, anisotropy,
                         change_of_curvature, eigenetropy]

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


def normalize_features(X):
    normalized = np.copy(X).T
    for i, feature_vector in enumerate(X.T):
        mean = np.mean(feature_vector)
        std = np.std(feature_vector)
        normalized_feature_vector = (feature_vector - mean) / std
        normalized[i] = normalized_feature_vector
    return normalized.T


def feature_selection(X):
    # Normalize features
    normalized = normalize_features(X)
    features = np.copy(normalized).T
    feature_set = []
    N_N_k = float(1 / 5)

    # Perform forward search to pick top 50% of features
    while len(feature_set) < max(len(features) / 2, 4):
        feature_J_val = []

        for feature in features:
            sw = 0
            sb = 0
            # Calculate within and between class scatter matrix
            for i in range(5):
                class_features = feature[100 * i:100 * (i + 1)]
                if len(feature_set) == 0:
                    # Within class
                    cov = np.cov(class_features)
                    sw += (N_N_k * cov)
                    # Between class
                    sample_mean = np.mean([class_features])
                    total_mean = np.mean(feature)
                    sb += (N_N_k * ((sample_mean * total_mean) * (sample_mean * total_mean).T))
                else:
                    # Within class
                    current_features = np.array(feature_set)[:, 100 * i:100 * (i + 1)]
                    cov = np.cov(np.append(current_features, [class_features], axis=0))
                    sw += (N_N_k * cov)
                    # Between class
                    sample_means = np.array([np.mean(np.append(current_features, [class_features], axis=0), axis=1)])
                    total_means = np.array([np.mean(np.append(feature_set, [feature], axis=0), axis=1)])
                    sb += (N_N_k * ((sample_means * total_means) * (sample_means * total_means).T))

            if len(feature_set) == 0:
                # sb and sw will be floats
                feature_J_val.append(sb / sw)
            else:
                feature_J_val.append(np.trace(sb) / np.trace(sw))
        # Sort features based on best J value
        ind = np.argsort(feature_J_val)
        features = features[ind]
        # Select best feature and add to feature set
        best_feature = features[-1]
        feature_set.append(best_feature)
        features = np.delete(features, len(features) - 1, axis=0)

    # Perform PCA on selected features to reduce dimensions to 4
    pca = PCA(n_components=4)
    principal_components = pca.fit_transform(np.array(feature_set).T)
    # Return feature set
    return principal_components


def feature_visualization(X):
    """
    Visualize the features
        X: input features. This assumes classes are stored in a sequential manner
    """
    # define the labels and corresponding colors
    colors = ['firebrick', 'grey', 'darkorange', 'dodgerblue', 'olivedrab']
    labels = ['building', 'car', 'fence', 'pole', 'tree']
    features = ['height', 'root_density', 'hull_area_2d', 'shape_index', 'sum_of_eigenvalues', 'linearity', 'planarity',
                'sphericity', 'omnivariance', 'anisotropy', 'change_of_curvature', 'eigenetropy', 'num_points',
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


def SVM_classification(X, y, hyperparameters):
    """
    Conduct SVM classification
        X: features
        y: labels
    """

    C, kernel, degree, gamma, decision_function_shape = hyperparameters
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clf = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, decision_function_shape=decision_function_shape)
    clf.fit(X_train, y_train)
    y_preds = clf.predict(X_test)

    print_metrics(y_preds, y_test, "SVM")


def RF_classification(X, y, hyperparameters):
    """
    Conduct RF classification
        X: features
        y: labels
    """
    n_estimators, criterion, max_features, bootstrap, max_samples = hyperparameters

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)
    clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion, max_features=max_features,
                                 bootstrap=bootstrap, max_samples=max_samples)
    clf.fit(X_train, y_train)
    y_preds = clf.predict(X_test)

    print_metrics(y_preds, y_test, "RF")


def print_metrics(y_preds, y_test, classifier_label):

    print("Class labels: 'building', 'car', 'fence', 'pole', 'tree'")
    print(classifier_label, "f1:", f1_score(y_test, y_preds, average=None))
    print(classifier_label, "precision:", precision_score(y_test, y_preds, average=None))
    print(classifier_label, "recall:", recall_score(y_test, y_preds, average=None))

    print("\n", classifier_label, "overall accuracy: %5.2f" % accuracy_score(y_test, y_preds))
    print(classifier_label, "f1 (weighted):", f1_score(y_test, y_preds, average='weighted'))
    print(classifier_label, "precision (weighted):", precision_score(y_test, y_preds, average='weighted'))
    print(classifier_label, "recall (weighted):", recall_score(y_test, y_preds, average='weighted'))

    print("\n", classifier_label, "confusion matrix")
    conf = confusion_matrix(y_test, y_preds)
    print(conf)


def plot_learning_curve(num_samp_train, mean_train_error_list, mean_test_error_list,
                        std_train_error_list, std_test_error_list, model_type, runs):
    plt.figure(figsize=(8, 6))

    # Compute Confidence Intervals (95% CI)
    ci_train = 1.96 * (np.array(std_train_error_list) / np.sqrt(runs))
    ci_test = 1.96 * (np.array(std_test_error_list) / np.sqrt(runs))

    # Plot mean error rates
    plt.plot(num_samp_train, mean_train_error_list, marker='o', linestyle='-', color='g', label='Apparent error')
    plt.plot(num_samp_train, mean_test_error_list, marker='o', linestyle='-', color='b', label='True error')

    # Fill between for confidence intervals
    plt.fill_between(num_samp_train,
                     np.array(mean_train_error_list) - ci_train,
                     np.array(mean_train_error_list) + ci_train,
                     color='g', alpha=0.2)

    plt.fill_between(num_samp_train,
                     np.array(mean_test_error_list) - ci_test,
                     np.array(mean_test_error_list) + ci_test,
                     color='b', alpha=0.2)

    plt.xlabel("Number of training samples",fontsize=12)
    plt.ylabel("Classification error",fontsize=12)
    plt.title(f"{model_type} Learning curve (mean of {runs} runs)",fontsize=15, weight='bold')
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()


def learning_curve(X, y, steps, h, model_type='SVM', runs=100):
    floatlist = np.linspace(0.01, 0.99, steps)  # Ensure values stay between 0 and 1
    mean_train_error_list = []  # Store Apparent Error Rate (AER)
    mean_test_error_list = []   # Store True Error Rate (TER)
    std_train_error_list = []
    std_test_error_list = []
    num_samp_train = []

    for i in floatlist:
        train_errors = []
        test_errors = []

        for _ in range(runs):
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=i)  # Only specify train_size

            if len(X_train) < 5 or len(X_test) < 5:  # Ensure enough samples
                continue

            if model_type == "SVM":
                clf = svm.SVC(**h)
            elif model_type == "RF":
                clf = RandomForestClassifier(**h)
            else:
                raise ValueError("Unknown model type")

            clf.fit(X_train, y_train)

            # Compute training and test errors
            train_acc = accuracy_score(y_train, clf.predict(X_train))
            test_acc = accuracy_score(y_test, clf.predict(X_test))

            train_errors.append(1 - train_acc)  # Apparent Error Rate (AER)
            test_errors.append(1 - test_acc)    # True Error Rate (TER)

        if len(train_errors) == 0:  # Avoid division by zero
            continue

        # Compute mean error rates over multiple runs
        mean_train_error = np.mean(train_errors)
        mean_test_error = np.mean(test_errors)

        mean_train_error_list.append(mean_train_error)
        mean_test_error_list.append(mean_test_error)

        # Compute standard deviations over multiple runs
        std_train_error = np.std(train_errors)
        std_test_error = np.std(test_errors)

        std_train_error_list.append(std_train_error)
        std_test_error_list.append(std_test_error)

        num_samp_train.append(len(X_train))

        print(f"{model_type}: Train-Test split {i:.2f}:{1 - i:.2f} → "
              f"AER: {mean_train_error:.4f}, TER: {mean_test_error:.4f}")

    plot_learning_curve(num_samp_train, mean_train_error_list, mean_test_error_list,
                        std_train_error_list, std_test_error_list, model_type, runs)


def generate_SVM_hyperparameters(X, y):
    """
    Conduct SVM hyperparameter tuning
        X: features
        y: labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    Cs = [0.1, 1, 5, 10, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 750, 1000]
    kernels = ['linear', 'poly', 'rbf', 'sigmoid']
    degrees = [1, 2, 3, 4, 5, 6]
    gammas = ['scale', 'auto', 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    decision_function_shapes = ['ovr', 'ovo']

    best_score = 0
    best_C = None
    best_kernel = None
    best_degree = None
    best_gamma = None
    best_decision_function_shape = None

    for C in tqdm(Cs, total=len(Cs)):
        for kernel in kernels:
            for degree in degrees:
                for gamma in gammas:
                    for decision_function_shape in decision_function_shapes:
                        clf = svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma,
                                      decision_function_shape=decision_function_shape)
                        clf.fit(X_train, y_train)
                        y_preds = clf.predict(X_test)

                        score = f1_score(y_test, y_preds, average='weighted')
                        if score > best_score:
                            best_score = score
                            best_C = C
                            best_kernel = kernel
                            best_degree = degree
                            best_gamma = gamma
                            best_decision_function_shape = decision_function_shape

    print("Best Hyperparameters [C, kernel, degree, gamma, decision_function_shape]:", best_C, best_kernel, best_degree,
          best_gamma, best_decision_function_shape)
    print("Best F1 (weighted):", best_score)

    hyperparameters = {
        'C': best_C,
        'kernel': best_kernel,
        'degree': best_degree,
        'gamma': best_gamma,
        'decision_function_shape': best_decision_function_shape
    }

    return hyperparameters


def generate_RF_hyperparameters(X, y):
    """
    Conduct RF hyperparameter tuning
        X: features
        y: labels
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    n_estimators = [25, 50, 75, 100, 125, 150, 200, 250, 300]
    crirerions = ['gini', 'entropy', 'log_loss']
    max_features = ["sqrt", "log2", 1, 2, 3, 4]
    bootstraps = [True, False]
    max_samples = [None, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9]

    best_score = 0
    best_n_estimators = None
    best_criterion = None
    best_max_features = None
    best_bootstrap = None
    best_max_samples = None

    for n_estimator in tqdm(n_estimators, total=len(n_estimators)):
        for criterion in crirerions:
            for max_feature in max_features:
                for bootstrap in bootstraps:
                    for max_sample in max_samples:
                        if not bootstrap:
                            # If not bootstrapping, ensure max_sample is none
                            max_sample = None

                        clf = RandomForestClassifier(n_estimators=n_estimator, criterion=criterion,
                                                     max_features=max_feature, bootstrap=bootstrap,
                                                     max_samples=max_sample)
                        clf.fit(X_train, y_train)
                        y_preds = clf.predict(X_test)

                        score = f1_score(y_test, y_preds, average='weighted')
                        if score > best_score:
                            best_score = score
                            best_n_estimators = n_estimator
                            best_criterion = criterion
                            best_max_features = max_feature
                            best_bootstrap = bootstrap
                            best_max_samples = max_sample

                        if not bootstrap:
                            # If not bootstrapping, do not have to redo calculation for each potential value
                            break

    print("Best Hyperparameters [n_estimator, criterion, max_feature, bootstrap, max_sample]:", best_n_estimators,
          best_criterion, best_max_features, best_bootstrap, best_max_samples)
    print("Best F1 (weighted):", best_score)

    hyperparameters = {
        'n_estimators': best_n_estimators,
        'criterion': best_criterion,
        'max_features': best_max_features,
        'bootstrap': best_bootstrap,
        'max_samples': best_max_samples
    }

    return hyperparameters


if __name__ == '__main__':
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
    refined_X = feature_selection(X=X)

    # SVM classification
    print('\nStart SVM hyperparameter tuning')
    h = generate_SVM_hyperparameters(X=refined_X, y=y)

    #print('\nStart SVM classification')
    #SVM_classification(X=refined_X, y=y, hyperparameters=h)

    # RF classification
    #print('\nStart RF hyperparameter tuning')
    #h = generate_RF_hyperparameters(X=refined_X, y=y)

    #print('\nStart RF classification')
    #RF_classification(X=refined_X, y=y, hyperparameters=h)

    # RF learning curve
    learning_curve(refined_X, y, 100, h, model_type="SVM", runs=50)
