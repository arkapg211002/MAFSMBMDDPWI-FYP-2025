# [Run using GPU based files : Recommended]
```
The streamlit application code is made based on gpu based files
```
## Implementing RAG for faster generation of wellbeing isnights without using any LLM

### To use RAG based wellebing insight in GPU based runtime
```python
# Saving the embedding model for cuda device
from sentence_transformers import SentenceTransformer
import pickle

# Load model once
embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda") # change to cpu when in cpu runtime

# Save it to a pickle file remove gpu when in cpu
with open("rag_embedding_gpu.pkl", "wb") as f:
    pickle.dump(embedding_model, f)

print("✅ Saved embedding_model to pkl")

# --------------------------------------

# Storing global store again as the embedding has changed based on cuda

import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

# Load embedding model from pickle
with open("rag_embedding_gpu.pkl", "rb") as f:
    embedding_model = pickle.load(f)
    print("✅ Loaded embedding_model from pkl")

# Load updated instruction data
with open("instruction_data.json", "r") as file:
    data = json.load(file)

documents = [f"{item['instruction']} {item['input']}" for item in data]
outputs = [item["output"] for item in data]
print(f"🗂 Total Records: {len(documents)}")

# Build FAISS index
embeddings = np.array([embedding_model.encode(doc) for doc in documents]).astype("float32")
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print(f"📦 FAISS Index rebuilt with {index.ntotal} vectors")

# Save to .pkl
with open("global_store.pkl", "wb") as f:
    pickle.dump({"documents": documents, "outputs": outputs, "index": index}, f)
print("💾 Saved updated global_store.pkl")

```
