import numpy as np
import os
import pandas as pd
from typing import Dict


ORTHOPEDIA_DIR = "/mnt/dss/dssfs04/pn25ke/pn25ke-dss-0001/orthopedia"
INFECT_CSV = "dat_orthopedia_infection2.csv"
NOINFECT_CSV = "dat_orthopedia_noinfection2.csv"
CSV_SEP = ";"
TRAIN_SPLIT = 0.7

# just have 0.6 - 0.2 - 0.2 split


def return_train_val_test_dfs(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    fn = df["Fallnummer"].unique()
    np.random.shuffle(fn)
    num_train = int(len(fn) * TRAIN_SPLIT)
    num_val = (len(fn) - num_train) // 2
    return {
        "train": df[df["Fallnummer"].isin(fn[:num_train])],
        "val": df[df["Fallnummer"].isin(fn[num_train:num_train + num_val])],
        "test": df[df["Fallnummer"].isin(fn[num_train + num_val:])],
    }


def main():
    # read CSV files
    infect_df = pd.read_csv(os.path.join(ORTHOPEDIA_DIR, "csv_files", INFECT_CSV), sep=CSV_SEP)
    noinfect_df = pd.read_csv(os.path.join(ORTHOPEDIA_DIR, "csv_files", NOINFECT_CSV), sep=CSV_SEP)
    
    # remove overlapping patients from noinfect
    noinfect_df = noinfect_df[~noinfect_df["Fallnummer"].isin(infect_df["Fallnummer"])]

    # keep scanned slides only
    infect_df = infect_df[infect_df["scanned"] == True]
    noinfect_df = noinfect_df[noinfect_df["scanned"] == True]
    
    # keep H&E stains only
    infect_df = infect_df[infect_df["staining"] == "HE"]
    noinfect_df = noinfect_df[noinfect_df["staining"] == "HE"]
    
    # keep images we have in the container only
    infect_images = [item.replace(".svs", "") for item in os.listdir(os.path.join(ORTHOPEDIA_DIR, "infect"))]
    infect_df = infect_df[infect_df["imageID"].astype("int").astype(str).isin(infect_images)]
    noinfect_images = [item.replace(".svs", "") for item in os.listdir(os.path.join(ORTHOPEDIA_DIR, "noinfect"))]
    noinfect_df = noinfect_df[noinfect_df["imageID"].astype("int").astype(str).isin(noinfect_images)]
    dfs = {
        "infect": infect_df,
        "noinfect": noinfect_df,
    }

    # print how many images we have
    print(f"Number of infection images: {len(infect_df)}")
    print(f"Number of no infection images: {len(noinfect_df)}")

    for key in dfs:
        # divide into train, validation and test sets; save dataframes as CSV files
        # df_splits = return_train_val_test_dfs(dfs[key])

        while True: 
            df_splits = return_train_val_test_dfs(dfs[key])
            print("Split:", [len(df_splits[split]) / len(dfs[key]) for split in ["train", "val", "test"]])
            if input("Works? yes or no\n") == "yes":
                break
        
        for split in ["train", "val", "test"]:
            df_splits[split].to_csv(os.path.join(ORTHOPEDIA_DIR, "csv_files", f"{key}_{split}.csv"), index=False)
            
        # read dataframes and print the length as a sanity check
        for split in ["train", "val", "test"]:
            print(key, split, len(pd.read_csv(os.path.join(ORTHOPEDIA_DIR, "csv_files", f"{key}_{split}.csv"))), "samples")


if __name__ == "__main__":
    main()
