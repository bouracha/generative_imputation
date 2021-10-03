import numpy as np

def simulate_motion_occlusions(X, num_occlusions=10, folder_name=""):
    '''
    Function to randomly replace num_occlusions values with their average value in the train set.
    Assumes we've save the (t_n, f_n) features as datasets/train_averages.csv
    :param X: set of ground truth 3D joint positions (batch_size, f_n, t_n)
    :param num_occlusions: number of occlusions per pose (int)
    :param folder_name: model_path (str)
    :return: set of ground truth 3D joint positions each with num_occlusions replaced (batch_size, f_n, t_n)
    '''
    m, n, t = X.shape
    X_occluded = np.copy(X)

    rng = np.random.default_rng()
    occlude_mask = np.zeros((m, t*n))
    occlude_mask[:, :num_occlusions] = 1.0
    rng.shuffle(occlude_mask, axis=1)
    occlude_mask = occlude_mask.reshape(m, n, t)
    occlude_mask = occlude_mask.astype('bool')
    assert (np.sum(occlude_mask[0]) == num_occlusions)
    assert (np.sum(occlude_mask) == num_occlusions * m)

    path = str(folder_name) + "datasets/train_averages.csv"
    avg_features = np.loadtxt(path, delimiter=',')
    avg_features_repeated = np.repeat(avg_features.reshape(n, t)[None], m, axis=0)
    #avg_features_repeated = np.zeros((m, n, t))
    assert (avg_features_repeated.shape == X.shape)

    X_occluded[occlude_mask] = avg_features_repeated[occlude_mask]

    return X_occluded, occlude_mask

def add_noise(X, alpha=1.0):
    '''
    Function to add gaussian noise scaled by alpha
    :param X: set of ground truth 3D joint positions (batch_size, 96)
    :param alpha: scalinng factor for amount of noise (float)
    :return: set of ground truth 3D joint positions each with added noise (batch_size, 96)
    '''
    m, n, t = X.shape
    X_added_noise = np.copy(X)

    X_added_noise = X_added_noise + alpha * np.random.uniform(0, 1, (m, n, t))

    return X_added_noise