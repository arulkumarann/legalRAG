from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('all-mpnet-base-v2')
sentences = ["What is the capital of France?", "Paris is the capital of France.", "The weather is nice today."]
embeddings = model.encode(sentences)
print(util.cos_sim(embeddings[0], embeddings[1]))  # Should be high
print(util.cos_sim(embeddings[0], embeddings[2]))  # Should be low