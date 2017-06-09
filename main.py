import numpy as np
import pickle
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Activation, Embedding
from keras.layers import LSTM
import keras
import preproc
import train_utils

DATA_DIR = "data"

if __name__ == "__main__":
    # Prepare training data
    VOCAB_SIZE = 3000
    data, labels = pickle.load(open(DATA_DIR + "/train_data.pyObj", "rb"))
    labels = labels - 1
    freq_vec = data["freq"]
    stroke_vec = data["stroke"]

    np.random.seed(125655)
    rand_idx = np.random.permutation(range(len(labels)))
    train_idx = rand_idx[:int(len(rand_idx) * 0.6)]
    cv_idx = rand_idx[int(len(rand_idx) * 0.6):int(len(rand_idx) * 0.2)]
    test_idx = rand_idx[int(len(rand_idx) * 0.8):]
    text_mat = preproc.preproc_text_vec(data["text"], VOCAB_SIZE)
    
    # Model building
    # define input layers
    input_text = Input(shape=(None, ), name="input_text")
    input_freq = Input(shape=(4,), name="input_freq")
    input_stroke = Input(shape=(3,), name="input_stroke")

    # embedding layers for text
    embed = Embedding(VOCAB_SIZE, 100)(input_text)
    lstm = LSTM(40)(embed)

    # dense for freq/stroke
    input_fs = keras.layers.concatenate([input_freq, input_stroke])
    dense1 = Dense(3, activation="sigmoid")(input_fs)

    # concatenate sequence layers and other two inputs
    concat = keras.layers.concatenate([lstm, dense1])

    # connect to outputs
    grade_out = Dense(6, activation="softmax")(concat)

    model = Model(inputs = [input_text, input_freq, input_stroke], 
                outputs = [grade_out])

    model.compile(optimizer = 'rmsprop', 
                loss="categorical_crossentropy", 
                metrics = ["accuracy"])


    # Setting up logging function


    ## tensorboard logging
    tb_callback = keras.callbacks.TensorBoard("logs")
    tb_callback.set_model(model)

    # Training
    EPOCH = 1
    cost_vec = []
    acc_vec = []
    print("Training data: %d samples" % len(train_idx))

    for epoch_i in range(EPOCH):
        print(" --- Epoch {} --- ".format(epoch_i))
        counter = 0

        for sample_idx in train_idx:
            sample = text_mat[sample_idx]
            label = labels[sample_idx]
            freq = freq_vec[sample_idx]
            stroke = stroke_vec[sample_idx]
            ret = model.train_on_batch(
                    [np.array([sample]), np.array([freq]), np.array([stroke])], 
                    keras.utils.to_categorical(label, 6))
            cost_vec.append(ret[0])
            acc_vec.append(ret[1])
            
            global_iter = epoch_i * len(train_idx) + counter        
            
            mv_cost = np.mean(cost_vec[-10:])
            mv_acc = np.mean(acc_vec[-10:])
            if (global_iter + 1) % 10 == 0:
                print("validating model...")
                val_loss, val_acc = train_utils.test_model(
                    model, test_idx, data, labels, VOCAB_SIZE)
                print("Validation: loss: {:f}, acc: {:.2f}".format(val_loss, val_acc))
                train_utils.write_log(tb_callback, global_iter, 
                        ["loss", "accuracy", "val_loss", "val_accuracy"], 
                        [mv_cost, mv_acc, val_loss, val_acc])
                
            if counter >= 3:
                break   
            print("[% 4d] cost: %s, accuracy: %.2f" % (counter, mv_cost, mv_acc))
            counter += 1

    model.save("model.h5")