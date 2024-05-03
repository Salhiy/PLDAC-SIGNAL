import zipfile
import h5py
import io
import numpy as np
import matplotlib.pyplot as plt
import random


def read_h5py(zip_filename, reshaped=False):

    folder_name = 'dsp'
    use_channels = [0, 1, 2]

    X, Y, S = [], [], []

    with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            if file_info.filename.startswith(folder_name) and not file_info.is_dir():
                # Extract the file from the zip archive
                with zip_ref.open(file_info) as hdf5_file:
                    with h5py.File(io.BytesIO(hdf5_file.read()), 'r') as hdf5_data:
                      for use_channel in use_channels:
                        row = hdf5_data['ch{}'.format(use_channel)][()]
                        if (not reshaped):
                            X.append(row)
                        else:
                            row_reshaped = []
                            for index, x in enumerate(row):
                                row_reshaped.append(x.reshape(32, 32))
                            X.append(np.array(row_reshaped))
                        Y.append(hdf5_data['label'][()][0])
                        S.append(file_info.filename.replace("dsp/", "").split("_")[0])
    return np.array(X, dtype="object"), np.array(Y, dtype="object"), np.array(S, dtype="object")

def leave_one_out_crosse_split(X, Y, S, s):
  X_train, Y_train, X_test, Y_test = [], [], [], []
  for index, v in enumerate(X):
    if S[index] == s:
      X_test.append(v)
      Y_test.append(Y[index])
    else:
      X_train.append(v)
      Y_train.append(Y[index])
  return np.array(X_train), np.array(Y_train, dtype='int'), np.array(X_test), np.array(Y_test, dtype='int')

def leave_one_out_cross_score(X, Y, S, train_method, showError=False):

  predectid_labels = []
  accuracy = []
  Iterations = []

  subjects = np.unique(S)

  for i, s in enumerate(subjects):
    X_train, Y_train, X_test, Y_test = leave_one_out_crosse_split(X, Y, S, s)
    
    score = train_method(X_train, X_test, Y_train, Y_test, eval=False)
    accuracy.append(score)
    Iterations.append(i)

  if (showError):
    plt.plot(Iterations, accuracy)
    # Add labels and title
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title('Accuracy per Iteration')

  score = np.mean(np.array([accuracy]))

  return score

def time_shift(X, max_shift = 5):
    shifted_X = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        shift_value = np.random.randint(-max_shift, max_shift + 1)
        shifted_X[i] = np.roll(X[i], shift_value, axis=1)
        
    return shifted_X

def image_cutout(image, max_rectangles=5, max_size=(25, 15)):
    augmented_image = np.copy(image)
    height, width = image.shape[:2]
    
    for _ in range(max_rectangles):
        rect_height = np.random.randint(1, max_size[0] + 1)
        rect_width = np.random.randint(1, max_size[1] + 1)
        y = np.random.randint(0, height - rect_height + 1)
        x = np.random.randint(0, width - rect_width + 1)
        augmented_image[y:y+rect_height, x:x+rect_width] = 0
    
    return augmented_image

def data_augment(X, Y, S, pourcentage):
    X_augmented = []
    Y_augmented = []
    S_augmented = []

    slice = int(X.shape[0] * pourcentage)

    data_ziped = list(zip(X[:slice], Y[:slice], S[:slice]))
    np.random.shuffle(data_ziped)

    #time shift
    for z in data_ziped:
        X_augmented.append(time_shift(z[0]))
        Y_augmented.append(z[1])
        S_augmented.append(z[2])
        
    for i, x in enumerate(X_augmented):
        for j, y in enumerate(x):
            X_augmented[i][j] = image_cutout(y)
    return np.array(X_augmented, dtype="object"), np.array(Y_augmented, dtype="object"), np.array(S_augmented, dtype="object")

def min_max_scaling_3d(data, a=0, b=1):
    # Calculate min and max along the specified axis
    min_vals = np.min(data)
    max_vals = np.max(data)
    
    # Perform Min-Max scaling
    scaled_data = (data - min_vals) / (max_vals - min_vals) * (b - a) + a
    
    return scaled_data

def pre_process_data(X, Y, S, apply_cov=False):
  X_cov, X_log = [], []
  if (apply_cov):
      X_cov = apply_cov(X)
  
  #apply log and reshape
  for x in X:
      sub_X_cov = []
      for y in x:
          sub_X_cov.append(np.log(y + 1e-3))
      X_log.append(np.array(sub_X_cov, dtype=np.float32))
            
  N = np.array(X_log, dtype="object")
  scalled = []
  #apply min max truncation
  for x in N:
      scalled.append(min_max_scaling_3d(x))

  return np.array(scalled, dtype="object") ,np.ravel(Y).astype('int'), np.ravel(S).astype('int')

def train_test_split(X, Y, test_size=0.4, random_state=None):
    if random_state is not None:
        random.seed(random_state)
    
    combined_data = list(zip(X, Y))
    
    random.shuffle(combined_data)
    
    split_index = int(len(combined_data) * (1 - test_size))
    
    X_train = [sample[0] for sample in combined_data[:split_index]]
    Y_train = [sample[1] for sample in combined_data[:split_index]]
    X_test = [sample[0] for sample in combined_data[split_index:]]
    Y_test = [sample[1] for sample in combined_data[split_index:]]
    
    return X_train, X_test, Y_train, Y_test