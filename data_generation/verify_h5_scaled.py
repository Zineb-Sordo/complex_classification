import pandas as pd
import h5py
import tqdm

if __name__ == '__main__':

    file_path = "../knee/metadata_knee.csv"
    df = pd.read_csv(file_path)

    for i in tqdm.tqdm(range(len(df))):
        file = h5py.File(df.location.iloc[i])
        if 'sc_kspace_scaled' in file.keys():
            continue
        else:
            print(df.location.iloc[i])
