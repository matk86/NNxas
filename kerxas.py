
# coding: utf-8

# In[1]:

from keras.models import Sequential

model = Sequential()


# # 3 hidden layers
# 
# 1000 units
# 
# relu

# In[2]:

from keras.layers import Dense, Activation

model.add(Dense(units=1000, input_dim=100))
model.add(Activation('relu'))
model.add(Dense(units=1000))
model.add(Activation('relu'))
model.add(Dense(units=1000))
model.add(Activation('relu'))
model.add(Dense(units=1000))
model.add(Activation('relu'))
model.add(Dense(units=4))


# In[3]:

from keras import optimizers

adam = optimizers.Adam(lr=0.00001, beta_1=0.95, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=["mse",'accuracy'])


# In[4]:

import json
import numpy as np


def load_xas_data(data_file="/global/homes/k/kmathew/all_data_processed_100000.json"):
    X, Y = [], []
    with open(data_file) as f:
        all_data = json.load(f)
    for d in all_data:
        X.append(d[1]) # spectrum
        Y.append(d[0]) # site
    return np.array(X), np.array(Y)


# In[5]:

X, Y = load_xas_data()

print(X.shape, Y.shape)

def get_xas_data(num_training, num_validation, num_test):
        
    # sample the data
    n_total = num_training + num_validation + num_test
    n_tv = num_training + num_validation
    
    test_mask = np.zeros(n_total, dtype=bool)
    test_choice = np.random.choice(n_total, num_test, replace=False)
    test_mask[test_choice] = True
    
    X_test = X[test_mask]
    Y_test = Y[test_mask]
    
    X_tv = X[~test_mask]
    Y_tv = Y[~test_mask]
    
    val_mask = np.zeros(n_tv, dtype=bool)
    val_choice = np.random.choice(n_tv, num_validation, replace=False)
    val_mask[val_choice] = True
    
    X_val = X_tv[val_mask]
    Y_val = Y_tv[val_mask]
    
    X_train = X_tv[~val_mask]
    Y_train = Y_tv[~val_mask]

    # Normalize the data: subtract the mean image
    #mean_image = np.mean(X_train, axis=0)
    #X_train -= mean_image
    #X_val -= mean_image
    #X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, Y_train, X_val, Y_val, X_test, Y_test


# In[7]:

num_total = X.shape[0]
validation_precentage = 10
test_precentage = 10
num_epochs = 50


# In[8]:

num_test = int(num_total * test_precentage /100.)
num_validation = int(num_total * validation_precentage /100.)
num_training = num_total - num_validation - num_test

X_train, Y_train, X_val, Y_val, X_test, Y_test = get_xas_data(num_training, num_validation, num_test)

print('Train data shape: ', X_train.shape)
print('Train labels shape: ', Y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', Y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', Y_test.shape)


# In[9]:

# print random sample's Y
Y_train[np.random.choice(num_training)]


# In[ ]:

model.fit(X_train, Y_train, epochs=num_epochs, verbose=1, validation_data=(X_val, Y_val), batch_size=32)


# In[ ]:

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test loss:', score[0])
print('Test accuracy:', score[1])


# In[ ]:

nchoice = np.random.choice(num_test, 5)
np.round(model.predict(X_test[nchoice]))


# In[ ]:

Y_test[nchoice]


# In[ ]:




# In[ ]:




# In[ ]:




# Tensormol: 3 hidden layers, 1000 neirons, with relu
# adam optimizer, minibatch method with l2 loss minimization, learning rate=0.00001, learning momentum=0.95
# check: TFMolInstance.py
# 
# inverse problem: given spectrum, predict structure
# define structure(descriptor): [spacegroup, formula(or just absorbing atom symbol), a, b, c]
#                   --> 225 len vec + 110 len vec + 3

# keras.preprocessing.sequence.pad_sequences(sequences, maxlen=None, dtype='int32',
#     padding='pre', truncating='pre', value=0.)
# 
# Transform a list of num_samples sequences (lists of scalars) into a 2D Numpy array of shape (num_samples, num_timesteps). num_timesteps is either the maxlen argument if provided, or the length of the longest sequence otherwise. Sequences that are shorter than num_timesteps are padded with value at the end. Sequences longer than num_timesteps are truncated so that it fits the desired length. Position where padding or truncation happens is determined by padding or truncating, respectively.
# 
# #Generate dummy data
# import numpy as np
# data = np.random.random((1000, 100))
# labels = np.random.randint(2, size=(1000, 1))
# 
# #Train the model, iterating on the data in batches of 32 samples
# model.fit(data, labels, epochs=10, batch_size=32)

# In[ ]:



