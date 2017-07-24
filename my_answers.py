import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras
import string


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    if window_size > len(series) - 1:
        raise ValueError("window_size is too big")

    X = [series[i:i + window_size] for i in range(len(series) - window_size)]
    y = series[window_size:]

    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

    return X,y

# TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape=(window_size, 1)))
    model.add(Dense(1))

    return model

### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    # Get unique characters from the text
    unique_chars = set(text)

    # Get all lowercase ascii characters
    ascii = set(string.ascii_lowercase)

    atypical_chars = unique_chars - ascii - set(punctuation)

    # Replace all atypical symbols
    for c in atypical_chars:
        text = text.replace(c, ' ')

    return text

### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = [text[i:i + window_size] for i in range(0, len(text) - window_size, step_size)]
    outputs = [text[i] for i in range(window_size, len(text), step_size)]

    return inputs,outputs

# TODO build the required RNN model: 
# a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape=(window_size, num_chars)))
    model.add(Dense(num_chars, activation='softmax'))

    return model


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    ### load in and normalize the dataset
    dataset = np.loadtxt('datasets/normalized_apple_prices.csv')
    print(dataset.shape)

    odd_nums = np.array([1,3,5,7,9,11,13])
    X, y = window_transform_series(odd_nums, 2)

    assert (type(X).__name__ == 'ndarray')
    assert (type(y).__name__ == 'ndarray')
    assert (X.shape == (5, 2))
    assert (y.shape in [(5, 1), (5,)])

    # print out input/output pairs --> here input = X, corresponding output = y
    print('--- the input X will look like ----')
    print(X)

    print('--- the associated output y will look like ----')
    print(y)

    # window the data using your windowing function
    window_size = 7
    X, y = window_transform_series(series=dataset, window_size=window_size)

    train_test_split = int(np.ceil(2 * len(y) / float(3)))  # set the split point

    # partition the training set
    X_train = X[:train_test_split, :]
    y_train = y[:train_test_split]

    print(X_train.shape)
    print(y_train.shape)

    # keep the last chunk for testing
    X_test = X[train_test_split:, :]
    y_test = y[train_test_split:]

    # NOTE: to use keras's RNN LSTM module our input must be reshaped to [samples, window size, stepsize]
    X_train = np.asarray(np.reshape(X_train, (X_train.shape[0], window_size, 1)))
    X_test = np.asarray(np.reshape(X_test, (X_test.shape[0], window_size, 1)))

    # given - fix random seed - so we can all reproduce the same results on our default time series
    np.random.seed(0)

    model = build_part1_RNN(window_size)

    # build model using keras documentation recommended optimizer initialization
    optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.0)

    # compile the model
    model.compile(loss='mean_squared_error', optimizer=optimizer)

    # run your model!
    model.fit(X_train, y_train, epochs=1000, batch_size=50, verbose=0)

    # generate predictions for training
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # print out training and testing errors
    training_error = model.evaluate(X_train, y_train, verbose=0)
    print('training error = ' + str(training_error))

    testing_error = model.evaluate(X_test, y_test, verbose=0)
    print('testing error = ' + str(testing_error))

    # read in the text, transforming everything to lower case
    text = open('datasets/holmes.txt').read().lower()
    print('our original text has ' + str(len(text)) + ' characters')

    ### find and replace '\n' and '\r' symbols - replacing them
    text = text[1302:]
    text = text.replace('\n', ' ')  # replacing '\n' with '' simply removes the sequence
    text = text.replace('\r', ' ')

    text = cleaned_text(text)
    text = cleaned_text(text)

    # shorten any extra dead space created above
    text = text.replace('  ', ' ')

    print(text[:50])

    inputs, outputs = window_transform_text(text[:50], 5, 3)
    print(inputs)
    print(outputs)
