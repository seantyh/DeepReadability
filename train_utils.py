import keras
import tensorflow as tf
import preproc
import numpy as np


def write_log(cb, iter_idx, names, logs):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        cb.writer.add_summary(summary, iter_idx)
        cb.writer.flush()

def test_model(model, data_idx, data, labels, VOCAB_SIZE):
    text_mat = preproc.preproc_text_vec(data["text"], VOCAB_SIZE)
    freq_vec = data["freq"]
    stroke_vec = data["stroke"]  
    cost_vec = []
    acc_vec = []
    for sample_idx in data_idx:
        sample = text_mat[sample_idx]
        label = labels[sample_idx]
        freq = freq_vec[sample_idx]
        stroke = stroke_vec[sample_idx]
        ret = model.test_on_batch([np.array([sample]), np.array([freq]), np.array([stroke])], 
                keras.utils.to_categorical(label, 6))
        cost_vec.append(ret[0])
        acc_vec.append(ret[1])
                        
    return (np.mean(cost_vec), np.mean(acc_vec))