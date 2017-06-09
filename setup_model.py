from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Embedding
from keras.layers import LSTM

def setup_model(VOCAB_SIZE):
    # define input layers
    input_text = Input(shape=(None, ), name="input_text")
    input_freq = Input(shape=(4,), name="input_freq")
    input_stroke = Input(shape=(3,), name="input_stroke")

    # embedding layers for text
    embed = Embedding(VOCAB_SIZE, 20)(input_text)
    lstm = LSTM(30)(embed)

    # dense for freq/stroke
    input_fs = keras.layers.concatenate([input_freq, input_stroke])
    dense1 = Dense(3, activation="sigmoid")(input_fs)    

    # concatenate sequence layers and other two inputs
    concat = keras.layers.concatenate([lstm, dense1])
    
    # connect to outputs
    dense2 = Dense(20, activation="sigmoid")(concat)
    grade_out = Dense(6, activation="softmax")(dense2)

    model = Model(inputs = [input_text, input_freq, input_stroke], 
                outputs = [grade_out])

    model.compile(optimizer = 'rmsprop', 
                loss="categorical_crossentropy", 
                metrics = ["accuracy"])