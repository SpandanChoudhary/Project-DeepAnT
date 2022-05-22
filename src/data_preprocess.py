import os

os.chdir("D:\\Projects\\Project-DeepAnT")

# libraries
import warnings
import numpy as np
import pandas as pd
from utils import *
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")


def preprocess():
    w = 20
    p_w = 1
    n_features = 1

    # complete dataset
    data = pd.read_csv("input/kpi_data.csv", index_col="datetime")
    data.reset_index(inplace=True)
    data.drop(["KPI ID"], axis=1, inplace=True)

    # train-test split
    train_df = data[: int(0.8 * len(data))]
    test_df = data[int(0.8 * len(data)) :]

    # data scaling
    deepant_scaler = MinMaxScaler()
    train_df[["scaled_value"]] = deepant_scaler.fit_transform(train_df[["value"]])
    test_df[["scaled_value"]] = deepant_scaler.transform(test_df[["value"]])

    # define input sequence
    raw_sequence = list(train_df["scaled_value"])
    # split into samples
    batch_sample, batch_label = utility.train_split_sequence(
        data_sequence=raw_sequence, w=w, p_w=p_w
    )
    # need to convert batch into 3D tensor of the form [batch_size, input_seq_len, n_features]
    batch_sample = batch_sample.reshape(
        (batch_sample.shape[0], batch_sample.shape[1], n_features)
    )

    # config variable
    batch_sample_train = batch_sample[: int(0.8 * len(batch_sample))]
    batch_sample_vali = batch_sample[int(0.8 * len(batch_sample)) :]

    batch_label_train = batch_label[: int(0.8 * len(batch_label))]
    batch_label_vali = batch_label[int(0.8 * len(batch_label)) :]

    # np.save
    np.save("input/batch_sample_train", batch_sample_train)
    np.save("input/batch_sample_vali", batch_sample_vali)
    np.save("input/batch_label_train", batch_label_train)
    np.save("input/batch_label_vali", batch_label_vali)

    # saving train and test dataframe
    train_df.to_csv("input/kpi_train.csv", index=False)
    test_df.to_csv("input/kpi_test.csv", index=False)


if __name__ == "__main__":
    preprocess()
