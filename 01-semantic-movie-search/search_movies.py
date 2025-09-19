"""
Enhanced movie search from Pinecone vector store with multi-field search
"""

from sentence_transformers import SentenceTransformer
from pinecone import Pinecone
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ----- Config from .env -----
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "movie-search-index")
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
TOP_K = int(os.getenv("TOP_K_RESULTS", "5"))

# Validate required environment variables
if not PINECONE_API_KEY:
    raise ValueError("❌ PINECONE_API_KEY not found in .env file")

if not INDEX_NAME:
    raise ValueError("❌ PINECONE_INDEX_NAME not found in .env file")

print("✅ Environment variables loaded successfully")

# ----- Load model & Pinecone -----
print("🤖 Loading model and Pinecone…")
model = SentenceTransformer(MODEL_NAME)
print(f"📊 Using embedding model: {MODEL_NAME}")

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Connect to the index
try:
    index = pc.Index(INDEX_NAME)
    print(f"🔗 Connected to Pinecone index: {INDEX_NAME}")
    
    # Get index stats to verify connection
    stats = index.describe_index_stats()
    print(f"📐 Index dimension: {stats.dimension}")
    print(f"🔢 Total vectors: {stats.total_vector_count}")
    
except Exception as e:
    print(f"❌ Error connecting to Pinecone index: {e}")
    print("📋 Available indexes:")
    available_indexes = list(pc.list_indexes())
    if available_indexes:
        for idx in available_indexes:
            print(f"- {idx.name}")
    else:
        print("No indexes found")
    exit()

# ----- Search examples -----
print("\n💡 Search examples:")
print("- 'space adventure by Alfonso Cuarón'")
print("- 'sci-fi movies with Sandra Bullock'")
print("- 'thriller films from 2013'")
print("- 'Oscar winning space movies'")
print("- 'George Clooney astronaut film'")

# ----- Enter query -----
user_query = input("\n🎬 Enter a description of the movie you're looking for:\n> ")

if not user_query.strip():
    print("❌ Please enter a valid search query")
    exit()

# Convert to vector
print("🔍 Generating embedding and searching...")
query_embedding = model.encode(user_query, normalize_embeddings=True).tolist()

# Search in Pinecone
try:
    results = index.query(
        vector=query_embedding,
        top_k=TOP_K,
        include_values=False,
        include_metadata=True
    )
    
    # ----- Display matches -----
    if results.matches:
        print(f"\n🎉 Found {len(results.matches)} matches for: '{user_query}'\n")
        
        for i, match in enumerate(results.matches, 1):
            title = match.metadata.get("title", "Unknown Title")
            year = match.metadata.get("year", "N/A")
            genres = match.metadata.get("genres", "")
            directors = match.metadata.get("directors", "")
            cast = match.metadata.get("cast", "")
            plot = match.metadata.get("plot", "")
            
            print(f"{i}. 🎥 {title} ({year})")
            print(f"   ⭐ Similarity score: {match.score:.3f}")
            
            if directors:
                print(f"   👨‍💼 Directors: {directors}")
            if genres:
                print(f"   🎭 Genres: {genres}")
            if cast:
                cast_preview = ', '.join(cast.split(', ')[:3])
                if len(cast.split(', ')) > 3:
                    cast_preview += "..."
                print(f"   🌟 Cast: {cast_preview}")
            if plot:
                print(f"   📝 Plot: {plot}")
            
            print("-" * 80)
    else:
        print("\n❌ No matches found. Try a different search term.")
        
except Exception as e:
    print(f"❌ Error querying Pinecone: {e}")

# Show search tips
print("\n💡 Search Tips:")
print("- Include director names: 'movies by Alfonso Cuarón'")
print("- Mention actors: 'films with Sandra Bullock'")
print("- Specify genres: 'sci-fi thriller'")
print("- Include years: '2013 space movies'")
print("- Combine multiple criteria: 'space adventure by Cuarón from 2013'")