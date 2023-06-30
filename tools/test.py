import tensorflow_hub as hub
import tensorflow_text

# Load the pre-trained model from TensorFlow Hub
model_url = "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3"
model = hub.load(model_url)

# Define a list of sentences to encode
sentences = [
    "Hello, how are you?",
    "I'm doing great, thanks for asking.",
    "The quick brown fox jumped over the lazy dog.",
    "Le renard brun rapide saute par-dessus le chien paresseux."
]

# Encode the sentences using the Universal Sentence Encoder model
embeddings = model(sentences)

# Print the resulting embeddings
for i, embedding in enumerate(embeddings):
    print("Sentence ", i+1, " embedding: ", embedding.numpy())
