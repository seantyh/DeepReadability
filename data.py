import numpy as np
import pickle

def load_data(dir_path, seq_length = 200):
    stroke_vec = []
    freq_bracket = []
    text_mat = []
    labels = []
    freq_map = pickle.load(open("etc/chFreq.pickle", "rb"))
    sorted_ch = sorted(freq_map.keys(), key = freq_map.get(), reverse=True)
    vocab_map = {k: i+1 for k, i in enumerated(sorted_ch)}
    stroke_map = load_stroke_data("etc/Unihan_DictionaryLikeData.txt")

    for fpath in glob.glob("*.txt"):
        text = load_text(fpath)
        txt, freqs, stks = transform_document(text)
        text_map.append(txt)
        freq_bracket.append(freqs)
        strokes.append(stks)
        label_x = int(re.search("_G(\d)_", fpath).group(1))  
        labels.append(label_x)

    data = {"freq": np.array(freq_bracket), 
            "stroke": np.array(stroke_vec), 
            "text": np.array(text_mat)}
    return data, np.array(label)

def transform_document(text):
    v1 = transform_text(text, vocab_map, seq_length))
    v2 = get_freq_bracket(text, vocab_map)
    v3 = get_stroke_bracket(text, stroke_map)
          
    return v1, v2, v3

def transform_text(text, vocab_map, seq_length):
    text_vec = [0] * seq_length
    counter = -1
    for ch in text:        
        idx = vocab_map.get(ch, -1)
        if idx < 0: continue
        counter += 1
        text_vec[counter] = idx
        if counter == seq_length - 1:
            break
    return text_vec


def get_freq_bracket(text, vocab_map):
    rank_vec = [0, 0, 0]
    for ch in text:
        rank = vocab_map.get(ch, -1):
        if freq < 0: continue
        if rank > 0 and rank <= 1500:
            rank_vec[0] += 1
        elif rank > 1500 and rank <= 3000:
            rank_vec[1] += 1
        else:
            rank_vec[2] += 1
    return rank_vec

def get_stroke_bracket(text, stroke_map):
    stk_vec = [0,0,0]
    for ch in text:
        stk = stroke_map.get(ch, -1)
        if stk < 0: continue
        if stk > 0 and stk <= 8:
            stk_vec[0] += 1
        elif stk > 8 and stk <= 12:
            stk_vec[1] += 1
        else:
            stk_vec[2] += 1
    return stk_vec

def load_stroke_data(fpath):
    stroke_map = {}
    with open(fpath, "r", encoding="UTF-8") as fin:
        for ln in fin.readlines():
            if ln.startswith("#"): continue
            if ln.find("kTotalStrokes") < 0: continue
            toks = ln.split("\t")
            cp = toks[0].replace("U+", "")            
            if len(cp) > 4: continue
            ch = codecs.decode(cp.encode(), "hex").decode("UTF-16BE")
            stroke_map[ch] = toks[2].strip()
    return stroke_map
    