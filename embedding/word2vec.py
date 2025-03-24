from gensim.models import Word2Vec

def apply_word2vec(sentences):
    """
    apply_word2vec
    params: sentences -> 'tokenized_text'
    returns: word2vec model

    Access vectors from base_model.wv.vectors and base_model.wv.index_to_key
    """
    base_model = Word2Vec(vector_size=100, min_count=5)
    base_model.build_vocab(sentences)
    # base_model.train(sentences, total_examples=base_model.corpus_count, epochs=base_model.epochs) 

    print(f'TODO: Use word2vec model to embed words in lyrics')

    return base_model