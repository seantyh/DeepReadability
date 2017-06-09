import numpy as np
import pickle

import keras
import preproc
import train_utils

DATA_DIR = "data"
USE_SAVED = True

if __name__ == "__main__":
    # Prepare training data
    VOCAB_SIZE = 2500
    data, labels = pickle.load(open(DATA_DIR + "/train_data.pyObj", "rb"))
    labels = labels - 1
    freq_vec = data["freq"]
    stroke_vec = data["stroke"]

    np.random.seed(125655)
    rand_idx = np.random.permutation(range(len(labels)))
    train_idx = rand_idx[:int(len(rand_idx) * 0.8)]    
    test_idx = rand_idx[int(len(rand_idx) * 0.8):]
    text_mat = preproc.preproc_text_vec(data["text"], VOCAB_SIZE)
    
    # Model building
    if USE_SAVED:
        model = keras.models.load_model("model.h5")
    else:
        model = setup_model.setup_model(VOCAB_SIZE)


    # Setting up logging function


    ## tensorboard logging
    tb_callback = keras.callbacks.TensorBoard("logs")
    tb_callback.set_model(model)

    # Training
    EPOCH = 10
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
            freq = freq / np.sum(freq)
            stroke = stroke/np.sum(stroke)
            ret = model.train_on_batch(
                    [np.array([sample]), np.array([freq]), np.array([stroke])], 
                    keras.utils.to_categorical(label, 6))
            cost_vec.append(ret[0])
            acc_vec.append(ret[1])
            
            global_iter = epoch_i * len(train_idx) + counter        
            
            mv_cost = np.mean(cost_vec[-100:])
            mv_acc = np.mean(acc_vec[-100:])
            if (global_iter + 1) % 100 == 0:
                train_utils.write_log(tb_callback, global_iter, 
                        ["loss", "accuracy"], [mv_cost, mv_acc])

            if (global_iter + 1) % 500 == 0:                        
                print("validating model...")
                val_loss, val_acc = train_utils.test_model(
                    model, test_idx, data, labels, VOCAB_SIZE)
                print("Validation: loss: {:f}, acc: {:.2f}".format(val_loss, val_acc))
                train_utils.write_log(tb_callback, global_iter, 
                        ["val_loss", "val_accuracy"], [val_loss, val_acc])    

            # if counter >= 3:
            #     break   
            print("[% 4d] cost: %s, accuracy: %.2f" % (counter, mv_cost, mv_acc))
            counter += 1

    model.save("model.h5")