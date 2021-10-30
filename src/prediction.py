import os
os.chdir('D:\\Projects\\Project-DeepAnT')

# libraries
import warnings
import numpy as np
import pandas as pd
from utils import *
from tensorflow import keras

warnings.filterwarnings('ignore')

test_df = pd.read_csv('input/kpi_test.csv', index_col= 'datetime')
test_df['datetime'] = test_df.index
prediction_model = keras.models.load_model('models/deepant.h5')

def get_range_proba(predict, label, delay=7):
    
    ''' 
    Changing the predict values based on delay allowed
    # predict: (list)
    # label: (list) 
    # delay: (int) Delay allowed (default 7)
    '''

    predict_arr = np.array(predict)
    label_arr = np.array(label)

    splits = np.where(label_arr[1:] != label_arr[:-1])[0] + 1
    is_anomaly = label_arr[0] == 1
    new_predict_arr = np.array(predict_arr)

    pos = 0
    for sp in splits:

        if is_anomaly:
            if 1 in predict_arr[pos:min(pos + delay + 1, sp)]:
                new_predict_arr[pos: sp] = 1
            else:
                new_predict_arr[pos: sp] = 0

        is_anomaly = not is_anomaly
        pos = sp

    sp = len(label_arr)

    if is_anomaly:
        if 1 in predict_arr[pos: min(pos + delay + 1, sp)]:
            new_predict_arr[pos: sp] = 1
        else:
            new_predict_arr[pos: sp] = 0

    return new_predict_arr

def prediction():

    w = 20
    p_w = 1
    n_features = 1

    test_sample, test_label, test_sample_time, test_label_time = utility.test_split_sequence( 
                                                                test_sequence = test_df, 
                                                                w = w, p_w = p_w)

    test_sample = test_sample.reshape((test_sample.shape[0], test_sample.shape[1], n_features))
    result_label = prediction_model.predict(test_sample, verbose=1)

    test_df.drop(['datetime'], axis = 1, inplace = True)
  
    tmp_result_df = pd.DataFrame()

    tmp_result_df['datetime'] = test_label_time.flatten()
    tmp_result_df['timestamp_label'] = test_label.flatten()
    tmp_result_df['timestamp_prediction'] = result_label.flatten()
    tmp_result_df['Pred Error'] = abs(tmp_result_df['timestamp_label'] -tmp_result_df['timestamp_prediction']).to_numpy()

    deepant_result_df = pd.merge(test_df, tmp_result_df, on = 'datetime', how = 'left')
    deepant_result_df['datetime'] = pd.to_datetime(deepant_result_df['datetime'])

    # static threshold method
    deepant_error_threshold = deepant_result_df['Pred Error'].mean() + 3*deepant_result_df['Pred Error'].std()

    #dynamic threshold method
    deepant_window = 20
    deepant_std_coef = 3

    deepant_result_df['Window Mean'] = deepant_result_df['Pred Error'].rolling(window=deepant_window, min_periods=1).mean()
    deepant_result_df['Window Std'] = deepant_result_df['Pred Error'].rolling(window=deepant_window, min_periods=1).std()
    deepant_result_df['Upper Bound'] = deepant_result_df['Window Mean'] + deepant_std_coef*deepant_result_df['Window Std']
    deepant_result_df.drop(['Window Mean','Window Std'], axis = 1, inplace =True)

    deepant_result_df['Anomaly_Predict'] = deepant_result_df['Pred Error'] >= deepant_result_df['Upper Bound']
    deepant_result_df['Static Predict'] = deepant_result_df['Pred Error'] > deepant_error_threshold

    deepant_result_df.set_index(['datetime'], inplace = True)

    # get range_proba function
    modified_static = get_range_proba(deepant_result_df['Static Predict'].astype('int64'), deepant_result_df['label'])
    deepant_result_df['Modified Static'] = modified_static

    modified_dynamic = get_range_proba(deepant_result_df['Anomaly_Predict'].astype('int64'), deepant_result_df['label'])
    deepant_result_df['Modified Dynamic'] = modified_dynamic
    deepant_result_df.reset_index(inplace =True)

    print('Modified Static Predictions: ', deepant_result_df['Modified Static'].sum())
    print('Modified Dynamic Predictions: ', deepant_result_df['Modified Dynamic'].sum())

    deepant_result_df.to_csv('input/prediction.csv', index = False)

if __name__ == "__main__":

    prediction()