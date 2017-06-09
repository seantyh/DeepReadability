def preproc_text_vec(text_mat, vocab_size):
    preproc_mat = []
    for text in text_mat:
        preproc_mat.append(list(filter(lambda x: x < vocab_size, text)))
        
    return preproc_mat