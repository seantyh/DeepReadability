from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
from data import load_data

DATA_DIR = "data"

if __name__ == "__main__":
    data, label = load_data(DATA_DIR)
    model = Sequential()
    model.add(Embedding(3000, 200), input_length = 100)
    model.add(LSTM(40))
    model.add(Dense(6, activation="softmax"))

    model.compile(optimizer = 'rmsprop', 
                  loss="categorical_crossentropy", 
                  metrics = ["accuracy"])

    model.predict(np.array([[2]])).shape
    # model.fit(data, labels, epochs = 10, batch_size = 32)


