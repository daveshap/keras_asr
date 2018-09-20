from keras.models import Sequential
from keras.layers.core import Dense, Activation, Droupout
from keras.layers.recurrent import LSTM


def model_alpha():
    model = Sequential()
    model.add(LSTM(5, 300, return_sequences=True))
    model.add(LSTM(300, 500, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(500, 200, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(200, 3))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    return model


def model_beta():
    model = Sequential()
    model.add(LSTM(2, 300, return_sequences=False))
    model.add(Dense(300, 2))
    model.add(Activation("linear"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    return model


def model_chi():
    hidden_units = 13
    nb_classes = 10
    model = Sequential()
    model.add(LSTM(output_dim=hidden_units,
                   init='uniform',
                   inner_init='uniform',
                   forget_bias_init='one',
                   activation='tanh',
                   inner_activation='sigmoid',
                   input_shape=X_train.shape[1:]))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)
    return model