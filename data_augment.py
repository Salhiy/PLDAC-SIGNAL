import numpy as np

def time_shift(X, max_shift = 5):
    shifted_X = np.zeros_like(X)
    
    for i in range(X.shape[0]):
        shift_value = np.random.randint(-max_shift, max_shift + 1)
        shifted_X[i] = np.roll(X[i], shift_value, axis=1)
        
    return shifted_X

def time_permute(X):
    return np.random.permutation(X)


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
        X_augmented.append(time_permute(time_shift(z[0])))
        Y_augmented.append(z[1])
        S_augmented.append(z[2])
        
    for i, x in enumerate(X_augmented):
        for j, y in enumerate(x):
            X_augmented[i][j] = image_cutout(y)
            
    return np.array(X_augmented, dtype="object"), np.array(Y_augmented, dtype="object"), np.array(S_augmented, dtype="object")
