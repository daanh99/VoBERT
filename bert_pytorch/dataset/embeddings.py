import numpy as np


class Embeddings():

    def __init__(self, embeddings, embedding_size):
        self.embeddings = embeddings
        self.embeddings[0] = np.zeros(embedding_size)
        self.embeddings[3] = np.full([embedding_size], 3)
        self.embeddings[4] = np.full([embedding_size], 4)

    def embed_sequence(self, seq):
        for token in seq:
            if token not in self.embeddings:
                raise Exception(f"Token {token} not in embeddings matrix")
        return [self.embeddings[token] for token in seq]
