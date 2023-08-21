#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Read the datafile into Python

import pandas as pd
import numpy as np

filename = 'https://raw.githubusercontent.com/avemuganti/MLData/main/Metro_Interstate_Traffic_Volume.csv'


# In[2]:


# Class definitions

class RNN:
    
    def __init__(self, filename, header = True):
        self.raw_input = pd.read_csv(filename);
        self.input_weights = np.random.rand(16,16) * 2/3 - 1/3
        self.recurrent_weights = np.random.rand(16,16) * 2/3 - 1/3
        self.output_weights = np.random.rand(16) * 2/3 - 1/3
        print("RNN Created!")
    
    def preprocess(self, data_slice = 1): # Will take the data in df and process it.
        newLen = int(len(self.raw_input) * data_slice)
        self.processed_data = self.raw_input.head(newLen)
        self.processed_data.drop(['holiday', 'weather_description'], axis = 1, inplace = True)

        lastDate = 'NONE'
        self.processed_data['date_time'] = pd.to_datetime(self.processed_data['date_time'])
        self.processed_data.reset_index(drop=True, inplace = True)
        # Create all the weather columns.
        weatherList = ['Clouds', 'Clear', 'Rain', 'Drizzle', 'Mist', 'Haze', 'Fog', 'Thunderstorm', 'Snow', 'Squall', 'Smoke']
        for elem in weatherList:
            self.processed_data[elem] = 0
        lastDateTime = 0
        n = 0

        for i in range(len(self.processed_data)):
            curDateTime = self.processed_data.loc[i,'date_time']
            if curDateTime == lastDateTime:
                n += 1
            else:
                n = 0
            self.processed_data.loc[i - n, self.processed_data.loc[i, 'weather_main']] = 1
            lastDateTime = curDateTime
        self.processed_data = self.processed_data.drop_duplicates(subset=["date_time"], keep='first')
        self.processed_data.reset_index(drop=True, inplace = True)
        # We need to fill in missing values with the previous values (temp, traffic volume)
        lastTemp = 0
        lastVol = 0
        for i in range(len(self.processed_data)):
            currVol = self.processed_data.loc[i, 'traffic_volume']
            currTemp = self.processed_data.loc[i, 'temp']
            if currVol == 0:
                self.processed_data.loc[i, 'traffic_volume'] = lastVol
            if currTemp == 0:
                self.processed_data.loc[i, 'temp'] = lastTemp
            lastTemp = currTemp
            lastVol = currVol
        lastDate_time = str(self.processed_data.iloc[len(self.processed_data) - 1, 5])
        # Fill in all missing dates and times.
        idx = pd.date_range('10/2/12 09:00:00', lastDate_time, freq = '1h')
        self.processed_data.set_index('date_time', inplace = True, drop = True)
        self.processed_data = self.processed_data.reindex(idx)
        self.processed_data.fillna(method = 'ffill', inplace = True)
        self.processed_data.rename_axis('date_time', inplace = True)
        self.processed_data.reset_index(inplace = True)
        self.processed_data['date'] = pd.to_datetime(self.processed_data['date_time']).dt.date
        self.processed_data['hour'] = pd.to_datetime(self.processed_data['date_time']).dt.time
        self.processed_data['hour'] = self.processed_data['hour'].apply(lambda x: x.strftime('%H'))
        self.processed_data['hour'] = self.processed_data['hour'].apply(lambda x: int(x))
        self.processed_data.drop(['date_time', 'weather_main'], axis = 1, inplace = True)
        self.processed_data.sort_values(by=['date', 'hour'], inplace = True)
        self.processed_data.drop(['date'], axis = 1, inplace = True)
        self.processed_data.reset_index(inplace = True, drop = True)
        # Put the traffic volume column at the end.
        col_list = list(self.processed_data.columns)
        col_list[4] = 'hour'
        col_list[16] = 'traffic_volume'
        self.processed_data = self.processed_data.reindex(columns = col_list)
        print("Preprocessing Done!")
        
    def split_data(self, trainratio = 0.8):
        train_len = round(trainratio * len(self.processed_data))
        self.dftrain = self.processed_data.iloc[:train_len]
        self.dftest = self.processed_data.iloc[train_len:].reset_index(drop = True)
        print("Train/Test Split Done!")
    
    def relu(self, val):
        return val if val > 0 else 0
    
    def step(self, val): # Derivative of Relu
        return 1 if val >= 0 else 0
    
    def clip(self, val, clp_factor): # Clip gradient at +- 1.
        if val > clp_factor:
            return clp_factor
        if val < -1 * clp_factor:
            return -1 * clp_factor
        return val
    
    def normalizeErrorsVector(self, factor, vec):
        sm = 0
        for i in vec:
            sm += i
        if abs(sm) < factor:
            return vec
        for i in range(len(vec)):
            vec[i] = vec[i] * (factor / abs(sm))
        return vec
        
    def normalizeErrorsMatrix(self, factor, mat):
        sm = 0
        for i in mat:
            for j in i:
                sm += j
        if abs(sm) < factor:
            return mat
        for i in range(len(mat)):
            for j in range(len(mat[i])):
                mat[i][j] = mat[i][j] * (factor / abs(sm))
        return mat
    
    def trainingRMSE(self): # Returns the training accuracy for the model (RMSE).
        rv = np.vectorize(self.relu)
        trainSize = len(self.dftrain)
        sqErrorSum = 0
        examples = 0
        for start_idx in range(trainSize - self.rnnSize + 1): # Goes through all training examples.
            examples += 1
            # Each training example starts at a different input vector
            recurrentvector = np.zeros(16) # initial values of recurrent vector
            for i in range(self.rnnSize):
                # Get the correct input vector
                inputvector = self.dftrain.iloc[start_idx + i,:16].to_numpy() # initial values of input vector
                # Calculate the next recurrent vector
                recurrentnorelu = np.matmul(self.input_weights, inputvector) + np.matmul(self.recurrent_weights, recurrentvector)
                recurrentvector = rv(recurrentnorelu)            
            # now create the output scalar from the recurrent vector.
            output = self.relu(np.dot(recurrentvector, self.output_weights))    
            expected = self.dftrain.iloc[start_idx + self.rnnSize - 1, 16]
            sqErrorSum += (expected - output) ** 2
        rmse = (sqErrorSum / (examples)) ** 0.5
        return rmse
    
    def testRMSE(self): # Returns the training accuracy for the model (RMSE).
        rv = np.vectorize(self.relu)
        testSize = len(self.dftest)
        sqErrorSum = 0
        examples = 0
        for start_idx in range(testSize - self.rnnSize + 1): # Goes through all test cases.
            examples += 1
            # Each training example starts at a different input vector
            recurrentvector = np.zeros(16) # initial values of recurrent vector
            for i in range(self.rnnSize):
                # Get the correct input vector
                inputvector = self.dftest.iloc[start_idx + i,:16].to_numpy() # initial values of input vector
                # Calculate the next recurrent vector
                recurrentnorelu = np.matmul(self.input_weights, inputvector) + np.matmul(self.recurrent_weights, recurrentvector)
                recurrentvector = rv(recurrentnorelu)            
            # now create the output scalar from the recurrent vector.
            output = self.relu(np.dot(recurrentvector, self.output_weights))    
            expected = self.dftest.iloc[start_idx + self.rnnSize - 1, 16]
            sqErrorSum += (expected - output) ** 2
        rmse = (sqErrorSum / (examples)) ** 0.5
        return rmse
    
    def train(self, learning_rate = 0.01, sz = 2, epochs = 10, clp_factor = 0.1, error_norm = 0.1):
        self.rnnSize = sz
        trainSize = len(self.dftrain)
        rv = np.vectorize(self.relu)
        sv = np.vectorize(self.step)
        clp = np.vectorize(self.clip)
        for it in range(epochs):
            for start_idx in range(trainSize - sz + 1): # Goes through all training examples.
                inputs = list() # List of all input vectors for this training example, in order
                recurrents = list() # List of all recurrent vectors for this training example, in order
                # First, get one iteration of forward propagation done.

                # Each training example starts at a different input vector
                recurrentvector = np.zeros(16) # initial values of recurrent vector
                recurrents.append(recurrentvector)
                for i in range(self.rnnSize):
                    # Get the correct input vector
                    inputvector = self.dftrain.iloc[start_idx + i,:16].to_numpy() # initial values of input vector
                    inputs.append(inputvector)
                    # Calculate the next recurrent vector
                    recurrentnorelu = np.matmul(self.input_weights, inputvector) + np.matmul(self.recurrent_weights, recurrentvector)
                    recurrentvector = rv(recurrentnorelu)
                    recurrents.append(recurrentvector)            
                # now create the output scalar from the recurrent vector.
                output = rv(np.dot(recurrentvector, self.output_weights))        
                #DEBUG: 
                print("output:" + str(output))
                # Backpropagation: Training of neural net
                expected = self.dftrain.iloc[start_idx + self.rnnSize - 1, 16]
                #DEBUG: 
                print("expected:" + str(expected))
                deltaOutput = (expected - output) * sv(output)
                # Update weights of the output_weights vector
                out_weight_change = np.zeros(16) # New vector for change of output weights
                for i in range(16):
                    out_weight_change[i] += learning_rate * deltaOutput * recurrents[-1][i]
                deltas = list()
                rec_weight_change = np.zeros((16, 16))
                inp_weight_change = np.zeros((16, 16))
                for i in range(self.rnnSize):
                    delJ = np.zeros(16)
                    if i == 0:
                        for j in range(16):
                            sm = deltaOutput * self.output_weights[j]
                            delJ[j] = sv(recurrents[-1][j]) * sm 
                    else:
                        for j in range(16):
                            sm = 0
                            for k in range(16): # sum up deltak and wkj from downstream nodes.
                                sm += deltas[-1 * i][k] * self.recurrent_weights[k][j]
                            delJ[j] = sv(recurrents[-1 - i][j]) * sm
                    deltas.insert(0, delJ) #DelJ has been populated, stored.
                    for j in range(16):
                        for k in range(16):
                            rec_weight_change[j][k] += learning_rate * delJ[j] * recurrents[-2 - i][k]
                            inp_weight_change[j][k] += learning_rate * delJ[j] * inputs[-1 - i][k]
                # Update all the weights in the rnn.
                self.output_weights += clp(self.normalizeErrorsVector(error_norm, out_weight_change), clp_factor)
                self.recurrent_weights += clp(self.normalizeErrorsMatrix(error_norm, rec_weight_change), clp_factor)
                self.input_weights += clp(self.normalizeErrorsMatrix(error_norm, inp_weight_change), clp_factor)
            print("Epoch " + str(it + 1) + " Complete!")
            print(self.output_weights)
            print(self.recurrent_weights)
            print(self.input_weights)
        print("Network Training Finished!")


# In[3]:


# Preprocessing

import copy

network = RNN(filename)
network2 = copy.deepcopy(network)
network.preprocess(data_slice = 0.01)
network2.preprocess(data_slice = 0.01)
network.split_data()
network2.split_data()


# In[4]:


trainErr1 = list()
testErr1 = list()
trainErr2 = list()
testErr2 = list()
epochNos = [100, 200, 300, 400, 500]
# Network will be trained for rnn size of 12, network2 will be trained for rnn size of 6.

for i in range(5):
    network.train(sz = 12, learning_rate = 0.00000000001, epochs = 100, clp_factor = 0.001, error_norm = 0.01)
    trainErr1.append(network.trainingRMSE())
    print("\n***\nNetwork 1 Training RMSE for " + str(epochNos[i]) + " epochs: " + str(trainErr1[-1]))
    testErr1.append(network.testRMSE())
    print("\n***\nNetwork 1 Test RMSE for " + str(epochNos[i]) + " epochs: " + str(testErr1[-1]))
    network2.train(sz = 6, learning_rate = 0.00000000001, epochs = 100, clp_factor = 0.001, error_norm = 0.01)
    trainErr2.append(network2.trainingRMSE())
    print("\n***\nNetwork 2 Training RMSE for " + str(epochNos[i]) + " epochs: " + str(trainErr2[-1]))
    testErr2.append(network2.testRMSE())
    print("\n***\nNetwork 2 Test RMSE for " + str(epochNos[i]) + " epochs: " + str(testErr2[-1]))


# In[21]:


import matplotlib.pyplot as plt

from tabulate import tabulate

# Create the table lists
netSize12 = list()
netSize6 = list()

for i in range(5):
    tup = list()
    tup.append(epochNos[i])
    tup.append(trainErr1[i])
    tup.append(testErr1[i])
    netSize12.append(tup)
    tup = list()
    tup.append(epochNos[i])
    tup.append(trainErr2[i])
    tup.append(testErr2[i])
    netSize6.append(tup)
    
head = ['Epochs', 'Training RMSE', 'Test RMSE']

print(tabulate(netSize12, headers=head, tablefmt="grid"))

plt.plot(epochNos, trainErr1, color = 'red', label = 'Training Error')
plt.plot(epochNos, testErr1, color = 'blue', label = 'Test Error')
plt.title('Network Size 12: Error')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.show()

print(tabulate(netSize6, headers=head, tablefmt="grid"))

plt.plot(epochNos, trainErr2, color = 'red', label = 'Training Error')
plt.plot(epochNos, testErr2, color = 'blue', label = 'Test Error')
plt.title('Network Size 6: Error')
plt.xlabel('Epochs')
plt.ylabel('RMSE')
plt.legend()
plt.show()


# In[24]:





# In[ ]:




