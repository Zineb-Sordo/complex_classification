import pandas as pd
import h5py


if __name__ == '__main__':

    file_path = "./metadata_knee.csv"
    df = pd.read_csv(file_path)

    for i in range(len(df)):
        file = h5py.File(df.location.iloc[i])
        if 'sc_kspace_scaled' in file.keys():
            continue
        else:
            print(df.location.iloc[i])
