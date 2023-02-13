
import pandas as pd
import numpy as np
import progressbar

"""# Pre-processing

#### Load Data
"""

control_raw_data = pd.read_csv('training_data/control.csv', header=None)
clench_raw_data = pd.read_csv('training_data/train.csv', header=None)

control_raw_data.drop(control_raw_data.columns[len(control_raw_data.columns ) -1], axis=1, inplace=True)
clench_raw_data.drop(clench_raw_data.columns[len(clench_raw_data.columns ) -1], axis=1, inplace=True)

"""#### Normalize"""

def normalize(raw_data):
    zeroed = raw_data.sub((raw_data.max() - raw_data.min() ) /2 + raw_data.min())
    scaled = zeroed.div((raw_data.max() - raw_data.min() ) /2)
    return scaled

control_data_preprocessed = normalize(control_raw_data)
clench_data_preprocessed = normalize(clench_raw_data)

"""#### Add Classes & Split into test/train/verify data"""

control_data_labeled = control_data_preprocessed.copy()
clench_data_labeled = clench_data_preprocessed.copy()

control_data_labeled['no_clench'] = 1
control_data_labeled['clench'] = 0

clench_data_labeled['no_clench'] = 0
clench_data_labeled['clench'] = 1

all_data = clench_data_labeled.append(control_data_labeled, ignore_index=True)
all_data = all_data.sample(frac=1, ignore_index=True)
all_data = all_data.sample(frac=1)

per_train = 0.6
per_test = 0.15
N = all_data.shape[0]

train_data  = all_data[:int(per_train *N)]
test_data   = all_data[int(per_train *N):int((per_train +per_test ) *N)]
verify_data = all_data[int((per_train +per_test ) *N):]

train_data.reset_index(inplace=True, drop=True)
test_data.reset_index(inplace=True, drop=True)
verify_data.reset_index(inplace=True, drop=True)

print(f'train: {train_data.shape}, test: {test_data.shape}, verify: {verify_data.shape}')

"""# Feature Extraction

#### Train AutoEncoder
"""

import progressbar
from keras.layers import Input, Dense, LSTM, RepeatVector, LeakyReLU, Dropout
from keras.models import Model
import numpy as np

latent_dim = 20
input_cols = 99

# build AutoEncoder
inputs = Input(shape=(input_cols,))
encoded = Dense(latent_dim)(inputs)
decoded = Dense(input_cols, name='reconstructed_layer')(encoded)

autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)
autoencoder.compile(optimizer='adam', loss='mse')
# autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

class_columns = ['no_clench', 'clench']
y_train_enc = train_data[class_columns]
x_train_enc = train_data.loc[:, ~train_data.columns.isin(class_columns)]
y_test_enc = test_data[class_columns]
x_test_enc = test_data.loc[:, ~test_data.columns.isin(class_columns)]

history = autoencoder.fit( x_train_enc, x_train_enc,
                           # epochs=20,
                           epochs=200,
                           # batch_size=256,
                           batch_size=200,
                           shuffle=False,
                           validation_data=(x_test_enc, x_test_enc),
                           verbose=1)

encoder.predict(x_train_enc)


def hjorth_complexity(X):
    Xp = deriv(X)
    if variance(X) == 0:
        raise Exception("Hjorth Mobility of signal is zero, will result in infinite Hjorth Complexity")
    return hjorth_mobility(Xp ) /hjorth_mobility(X)


def hjorth_mobility(X):
    Xp = deriv(X)
    if variance(X) == 0:
        raise Exception("variance of signal is zero, will result in infinite Hjorth Mobility")
    return (variance(Xp ) /variance(X) ) *0.5

def deriv(X):
    Xp = np.zeros(shape=(X.shape[0 ] -1))
    for k in range(X.shape[0 ] -1):
        Xp[k] = (X[ k +1 ] -X[k])
    return Xp

def variance(X):
    return np.var(X)

def simple_square_integral(X):
    return np.square(X).sum()

def willison_amplitude(X ,threshold=0):
    WA = 0
    for k in range(X.shape[0 ] -1):
        if abs(X[ k +1 ] -X[k]) > threshold:
            WA += 1
    return WA /X.shape[0]

def root_mean_square(X):
    return (( 1 /X.shape[0] ) *np.square(X).sum() ) *0.5

def mean(X):
    return X.sum( ) /X.shape[0]

def slope_sign_changes(X ,threshold=0):
    SSC = 0
    for k in range(1 ,len(X ) -1):
        if (X[k ] -X[ k -1] ) *(X[k ] -X[ k +1]) > threshold:
            SSC += 1
    return SSC / X.shape[0]

def zero_crossing(X ,threshold=0):
    ZC = 0
    for k in range(len(X ) -1):
        if (X[k] > 0 and X[ k +1] < 0) or (X[k] < 0 and X[ k +1] > 0 and abs(X[k ] -X[ k +1] ) >threshold):
            ZC += 1
    return ZC

def waveform_length(X):
    WL = 0
    for i in range(len(X ) -1):
        WL += abs(X[ i +1 ] -X[i])
    return WL

def mean_absolute_value(X):
    return ( 1 /X.shape[0] ) *np.absolute(X).sum()

def extract_predefined_features(raw_data):
    output_data = None
    feature_funcs = [mean_absolute_value,
                     waveform_length,
                     zero_crossing,
                     slope_sign_changes,
                     mean,
                     root_mean_square,
                     willison_amplitude,
                     simple_square_integral,
                     variance,
                     hjorth_mobility,
                     hjorth_complexity]

    feature_funcs_cols = ['mean_absolute_value',
                          'waveform_length',
                          'zero_crossing',
                          'slope_sign_changes',
                          'mean',
                          'root_mean_square',
                          'willison_amplitude',
                          'simple_square_integral',
                          'variance',
                          'hjorth_mobility',
                          'hjorth_complexity']

    new_data = None

    for index, row in raw_data.iterrows():
        signal = row.to_numpy()
        predefined_features =[]
        for feature_func in feature_funcs:
            predefined_features += [feature_func(signal)]
        if new_data is None:
            new_data = predefined_features
        else:
            new_data = np.vstack((new_data, predefined_features))

    return pd.DataFrame(new_data, columns=feature_funcs_cols)


def extract_features(raw_data):
    class_columns = ['no_clench', 'clench']
    only_data = raw_data.loc[:, ~raw_data.columns.isin(class_columns)]
    only_classes = raw_data[class_columns]
    autoencoded_features = encoder.predict(only_data)
    autoencoded_df = pd.DataFrame(autoencoded_features)
    predef_features = extract_predefined_features(only_data)
    return autoencoded_df.join(predef_features).join(only_classes)
    # return raw_data


train_features = extract_features(train_data)
test_features = extract_features(test_data)
verify_features = extract_features(verify_data)

num_cols = train_features.shape[1] - 2

"""# Build Model"""

from keras.layers import Dropout
import random
from keras.models import Sequential, Model
from keras.layers import *

input_nn = Input(shape=(num_cols,))
hidden = Dense(num_cols, activation='relu', name='hidden1')(input_nn)
hidden = Dropout(0.1)(hidden)
hidden = Dense(num_cols, activation='relu', name='hidden3')(hidden)
output_nn = Dense(2, activation='sigmoid', name='predicted_class')(hidden)

neural_network = Model(input_nn, output_nn)
neural_network.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

class_columns = ['no_clench', 'clench']
y_train = train_features[class_columns]
x_train = train_features.loc[:, ~train_features.columns.isin(class_columns)]
y_test = test_features[class_columns]
x_test = test_features.loc[:, ~test_features.columns.isin(class_columns)]

history = neural_network.fit(x_train, y_train,
                             epochs=50,
                             # batch_size=256,
                             batch_size=250,
                             shuffle=True,
                             validation_data=(x_test, y_test),
                             verbose=1)

print("history = " + str(history.history))

"""# Train Model

# Verify Model
"""

class_columns = ['no_clench', 'clench']
y_verify = verify_features[class_columns].to_numpy()
x_verify = verify_features.loc[:, ~verify_features.columns.isin(class_columns)]

print(f'verify shape: {verify_features.shape}')

predicts = neural_network.predict(x_verify)
prediction_col = np.zeros(shape=(predicts.shape[0], 1))
num_correct = 0
confusion_matrix = np.zeros(shape=(2, 2))

for i in range(predicts.shape[0]):
    is_clutched_predict = 0
    if predicts[i][0] < predicts[i][1]:
        is_clutched_predict = 1

    is_clutched_real = 0
    if y_verify[i][0] < y_verify[i][1]:
        is_clutched_real = 1

    confusion_matrix[is_clutched_real, is_clutched_predict] += 1

    if is_clutched_predict == is_clutched_real:
        num_correct += 1

print(confusion_matrix)
print(f"testing accuracy = {100 * (num_correct / predicts.shape[0])}%")

"""# Saving/Loading model to disk"""

# serialize model to JSON
model_json = neural_network.to_json()
with open("model/model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
neural_network.save_weights("model/model.h5")
print("Saved model to disk")

# # load json and create model
# json_file = open('model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")