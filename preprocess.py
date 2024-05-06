import numpy as np
import random

def min_max_scaling_3d(data, a=0, b=1):
    # Calculate min and max along the specified axis
    min_vals = np.min(data)
    max_vals = np.max(data)
    
    # Perform Min-Max scaling
    scaled_data = (data - min_vals) / (max_vals - min_vals) * (b - a) + a
    
    return scaled_data

def pre_process_data(X, Y, S):
  X_log = []
  
  #apply log and reshape
  for x in X:
      sub_X_log = []
      for y in x:
          sub_X_log.append(np.log(y + 1e-3))
      X_log.append(np.array(sub_X_log, dtype=np.float32))
            
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