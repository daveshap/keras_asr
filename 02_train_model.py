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


def model_delta(n_input, n_output, n_units):
    # define training encoder
    encoder_inputs = Input(shape=(None, n_input))
    encoder = LSTM(n_units, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(n_units, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(n_units,))
    decoder_state_input_c = Input(shape=(n_units,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    return model, encoder_model, decoder_model