import zipfile
import h5py
import io
import numpy as np


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