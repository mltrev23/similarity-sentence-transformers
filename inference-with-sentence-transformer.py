from sentence_transformers import SentenceTransformer
import torch
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(sentences)
print(embeddings)

loss_fn = torch.nn.CrossEntropyLoss()
print(loss_fn(torch.from_numpy(embeddings[0]), torch.from_numpy(embeddings[1])))