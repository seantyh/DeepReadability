import numpy as np
import pickle
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding
from keras.layers import LSTM
import keras
import preproc

DATA_DIR = "data"

if __name__ == "__main__":
    VOCAB_SIZE = 3000
    data, labels = pickle.load(open(DATA_DIR + "/train_data.pyObj", "rb"))
    labels = labels - 1

    np.random.seed(125655)
    rand_idx = np.random.permutation(range(len(labels)))
    train_idx = rand_idx[:int(len(rand_idx) * 0.6)]
    cv_idx = rand_idx[int(len(rand_idx) * 0.6):int(len(rand_idx) * 0.2)]
    test_idx = rand_idx[int(len(rand_idx) * 0.8):]
    text_mat = preproc.preproc_text_vec(data["text"], VOCAB_SIZE)
    
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, 100))
    model.add(LSTM(40))
    model.add(Dense(6, activation="softmax"))

    model.compile(optimizer = 'rmsprop', 
                  loss="categorical_crossentropy", 
                  metrics = ["accuracy"])

    counter = 0
    cost_vec = []
    acc_vec = []
    print("Training data: %d samples" % len(train_idx))

    EPOCH = 10
    for epoch_i in range(EPOCH):
        print(" --- Epoch {} --- ".format(epoch_i))
        counter = 0

        for sample_idx in train_idx:
            sample = text_mat[sample_idx]
            label = labels[sample_idx]
            ret = model.train_on_batch(np.array([sample]), keras.utils.to_categorical(label, 6))
            cost_vec.append(ret[0])
            acc_vec.append(ret[1])

            print("[% 4d] cost: %s, accuracy: %.2f" % \
                (counter, np.mean(cost_vec[-10:]), np.mean(acc_vec[-10:])))   
            counter += 1
            # if counter > 5:
            #     break   
    # model.fit(data, labels, epochs = 10, batch_size = 32)


