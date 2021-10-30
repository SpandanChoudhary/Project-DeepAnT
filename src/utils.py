# libraries 
from numpy import array

class utility():

    @staticmethod
    def train_split_sequence(data_sequence, w, p_w):
    
        print('Sliding Window Size: ', w)
        print('Prediction Window Size: ', p_w)
        print('-----------------------------')
    
        x, y = [], []
    
        for i in range(len(data_sequence)):
    
            # find the end index 
            end_index = i + w
            output_end_index = end_index + p_w
        
            # check for length
            if output_end_index > len(data_sequence):
                break
        
            # gather input and output parts of the pattern
            seq_x = data_sequence[i:end_index]
            seq_y = data_sequence[end_index:output_end_index]
        
            x.append(seq_x)
            y.append(seq_y)
        
        return array(x), array(y)

    @staticmethod
    def test_split_sequence(test_sequence, w, p_w):
    
        print('Sliding Window Size: ', w)
        print('Prediction Window Size: ', p_w)
        print('-----------------------------')
   
        x_test, y_test, sample_time_test, time_test = [],[],[],[]
    
        for i in range(len(test_sequence)):
        
            # find the end of this pattern
            end_ix = i + w
            out_end_ix = end_ix + p_w
        
            # check if we are beyond the sequence
            if out_end_ix > len(test_sequence):
                break
            # gather input and output parts of the pattern
            seq_x = test_sequence['scaled_value'][i:end_ix] 
            seq_sample_time = test_sequence['datetime'][i:end_ix] 
        
            seq_y = test_sequence['scaled_value'][end_ix:out_end_ix]
            seq_time = test_sequence['datetime'][end_ix :out_end_ix]
        
            x_test.append(seq_x)
            sample_time_test.append(seq_sample_time)
        
            y_test.append(seq_y)
            time_test.append(seq_time)
        
        return array(x_test), array(y_test),array(sample_time_test), array(time_test)