from sentence_transformers import SentenceTransformer


def bert_classification_embedding(sentences, torch_device):
    sentences = list(sentences)
    print(f"Encoding {len(sentences)} log description sentences with sentence BERT")
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    # sbert_model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v1')
    sentence_embeddings = sbert_model.encode(sentences, batch_size=64, device=torch_device, show_progress_bar=True,
                                             convert_to_numpy=True)
    print("Done encoding sentences with sentence BERT into vectors of dim " + str(sentence_embeddings[0].shape))
    # print(sentence_embeddings)

    return sentence_embeddings
