from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import time
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ----- Config from .env -----
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "movie-search-index")
EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CLOUD = os.getenv("PINECONE_CLOUD", "aws")
REGION = os.getenv("PINECONE_REGION", "us-east-1")
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB_NAME", "movies")
COLLECTION_NAME = os.getenv("MONGO_COLLECTION_NAME", "movies")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "100"))

# Validate required environment variables
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in .env file")

print("✅ Environment variables loaded successfully")

# Get the dimension of your embedding model
model = SentenceTransformer(EMBED_MODEL)
test_embedding = model.encode(["test"])
EMBEDDING_DIMENSION = len(test_embedding[0])
print(f"Embedding model dimension: {EMBEDDING_DIMENSION}")

# Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Create index if not exists with the correct dimension
existing_indexes = [index.name for index in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    print(f"Creating index '{INDEX_NAME}' with dimension {EMBEDDING_DIMENSION}...")
    pc.create_index(
        name=INDEX_NAME,
        dimension=EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=CLOUD,
            region=REGION
        )
    )
    # Wait for index to be ready
    print("Waiting for index to be ready...")
    time.sleep(30)  # Adjust as needed based on index creation time

# Get Index object
index = pc.Index(name=INDEX_NAME)

# MongoDB connection
mongo = MongoClient(MONGO_URI)
movies = mongo[DB_NAME][COLLECTION_NAME]

# Get total count for progress tracking
total_movies = movies.count_documents({})
print(f"Found {total_movies} movies to index")

# Indexing movies
batch = []
count = 0

for m in movies.find({}, {"_id":1, "title":1, "fullplot":1, "year":1}):
    text = f"{m.get('title','')} {m.get('fullplot','')}"
    embedding = model.encode(text, normalize_embeddings=True).tolist()

    batch.append({
        "id": str(m["_id"]),
        "values": embedding,
        "metadata": {"title": m.get("title"), "year": m.get("year"), "text": text}
    })

    if len(batch) >= BATCH_SIZE:
        index.upsert(batch)
        count += len(batch)
        print(f"✅ Indexed {count}/{total_movies} movies so far…")
        batch = []

# Final batch
if batch:
    index.upsert(batch)
    count += len(batch)
    print(f"✅ Indexed {count} movies total")

print(f"🎉 Finished indexing movies into Pinecone index '{INDEX_NAME}'")