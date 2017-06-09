import keras
import preproc
from data import *
import numpy as np

class Decoder:
    def __init__(self, model_path):
        self.model = keras.models.load_model(model_path)
        freq_map = pickle.load(open("etc/chFreq.pickle", "rb"))
        sorted_ch = sorted(freq_map.keys(), key = freq_map.get, reverse=True)
        vocab_map = {k: i+1 for i, k in enumerate(sorted_ch)}
        stroke_map = load_stroke_data("etc/Unihan_DictionaryLikeData.txt")
        self.params = {"vocab_map": vocab_map, "stroke_map": stroke_map, "seq_length": -1,
                  "VOCAB_SIZE": 2500}
        
    def predict_text(self, text):        
        txt, freqs, stks = transform_document(text, self.params)
        text_mat = preproc.preproc_text_vec([txt], self.params["VOCAB_SIZE"])
        ret = self.model.predict_on_batch([np.array([text_mat[0]]), np.array([freqs]), np.array([stks])])        
                            
        return np.argmax(ret)
    
    def get_stroke_vec(self, text):
        txt, freqs, stks = transform_document(text, self.params)
        return stks

    def get_freq_vec(self, text):
        txt, freqs, stks = transform_document(text, self.params)
        return freqs