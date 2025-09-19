## Semantic Movie Search (Pinecone + MongoDB)

This mini-project lets you index a MongoDB movies dataset into a Pinecone vector index and perform semantic search over it using SentenceTransformers.

### Prerequisites
- **Python**: 3.10+
- **uv**: Fast Python package/deps manager. Install: `pip install uv` or see `https://docs.astral.sh/uv/`
- **Pinecone account**: Create an account and get an API key. Choose a serverless Cloud/Region supported by your account.
- **MongoDB**: Local MongoDB or MongoDB Atlas. You need a database (default: `movies`) and a collection (default: `movies`) containing the movies JSON from this folder.

### Environment variables (.env)
Create a `.env` file in the project root with the following configuration. Adjust values as needed.

```env
# Pinecone
PINECONE_API_KEY=your_pinecone_api_key
PINECONE_INDEX_NAME=movie-search-index
PINECONE_CLOUD=aws         # or gcp, per your Pinecone project
PINECONE_REGION=us-east-1  # pick a region supported by your Pinecone project

# Embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2

# MongoDB
MONGO_URI=mongodb://localhost:27017
MONGO_DB_NAME=movies
MONGO_COLLECTION_NAME=movies

# App tuning
BATCH_SIZE=100
TOP_K_RESULTS=5
```

Notes:
- `load_movies.py` will create the Pinecone index if it does not exist, using the embedding dimension of the configured model.
- `search_movies.py` expects the index to exist and contain vectors.

### Setup with uv (Windows PowerShell examples)
Run the following from the project root (`01-semantic-movie-search`).

```powershell
# 1) Create a virtual environment managed by uv
uv venv

# 2) Install dependencies into the venv
uv pip install pymongo sentence-transformers pinecone-client python-dotenv

# Optional: activate the venv for your shell session
. .\.venv\Scripts\Activate.ps1
```

Alternatively, you can skip activation and use `uv run` to execute scripts directly within the venv:

```powershell
uv run python --version
```

### Import the movies JSON into MongoDB
Ensure your movies JSON file from this folder is imported into MongoDB. Replace the file name below with the actual file name if different.

Local MongoDB example:
```powershell
mongoimport --uri "mongodb://localhost:27017" `
  --db movies `
  --collection movies `
  --file .\movies.json `
  --jsonArray
```

MongoDB Atlas example:
```powershell
mongoimport --uri "mongodb+srv://<user>:<password>@<cluster>/?retryWrites=true&w=majority" `
  --db movies `
  --collection movies `
  --file .\movies.json `
  --jsonArray
```

Verify the import (optional, in a MongoDB shell or Compass):
- Database: `movies`
- Collection: `movies`
- Documents present

### Run: Index movies into Pinecone
This reads documents from MongoDB, embeds them, and upserts to Pinecone.

Using an activated venv:
```powershell
python .\load_movies.py
```

Without activation (uses uv):
```powershell
uv run python .\load_movies.py
```

Expected behavior:
- Validates `.env` and loads the embedding model
- Creates the Pinecone index if missing (serverless spec using your cloud/region)
- Batches and upserts embeddings to the index

### Run: Semantic search
This script prompts for a natural-language description and returns similar movies from Pinecone.

With activated venv:
```powershell
python .\search_movies.py
```

Without activation (uses uv):
```powershell
uv run python .\search_movies.py
```

### Troubleshooting
- If you see "PINECONE_API_KEY not found", ensure `.env` exists and is in the project root.
- If Pinecone connection fails, verify `PINECONE_CLOUD`/`PINECONE_REGION` match your Pinecone project settings and that the index name is correct.
- If no search results, verify that `load_movies.py` completed and that your MongoDB collection has meaningful `title`/`fullplot` fields.


