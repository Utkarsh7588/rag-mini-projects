import os
import re
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

class StoryIngestor:
    def __init__(self):
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        # Initialize clients
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
    def get_embedding(self, text):
        """Get embedding for text using OpenAI"""
        response = self.openai_client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    
    def create_index(self):
        """Create Pinecone index if it doesn't exist"""
        if self.index_name not in self.pc.list_indexes().names():
            print(f"Creating index {self.index_name}...")
            self.pc.create_index(
                name=self.index_name,
                dimension=1536,  # OpenAI embedding dimension
                metric="cosine",
                spec=ServerlessSpec(
                    cloud=os.getenv("PINECONE_CLOUD"),
                    region=os.getenv("PINECONE_REGION")
                )
            )
            # Wait for index to be ready
            time.sleep(60)
        
        return self.pc.Index(self.index_name)
    
    def read_and_preprocess_story(self, file_path):
        """Read and preprocess the story file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Preprocessing steps
            content = self.clean_text(content)
            return content
            
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return None
        except Exception as e:
            print(f"Error reading file: {e}")
            return None
    
    def clean_text(self, text):
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep punctuation
        text = re.sub(r'[^\w\s.,!?;:\'"-]', '', text)
        return text.strip()
    
    def create_document_chunks(self, text, chunk_size=800, overlap=150):
        """Create document chunks with semantic boundaries"""
        # Split into sentences first for better chunking
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size, save current chunk
            if current_length + sentence_length > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep overlap by carrying over last few sentences
                overlap_sentences = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk[-1:]
                current_chunk = overlap_sentences
                current_length = sum(len(s) for s in overlap_sentences) + len(overlap_sentences) - 1
                
                # Add current sentence to new chunk
                current_chunk.append(sentence)
                current_length += sentence_length + 1
            else:
                current_chunk.append(sentence)
                current_length += sentence_length + 1
        
        # Add the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def create_documents_with_metadata(self, chunks, source_file):
        """Create documents with metadata"""
        documents = []
        for i, chunk in enumerate(chunks):
            documents.append({
                'id': f"chunk_{i}",
                'text': chunk,
                'metadata': {
                    'source': source_file,
                    'chunk_index': i,
                    'chunk_size': len(chunk)
                }
            })
        return documents
    
    def ingest_story(self, file_path="story.txt"):
        """Ingest and embed the story file"""
        # Read and preprocess the story
        story_text = self.read_and_preprocess_story(file_path)
        if not story_text:
            print("Failed to read story file.")
            return
        
        print(f"Original story length: {len(story_text)} characters")
        
        # Create document chunks
        chunks = self.create_document_chunks(story_text)
        print(f"Split into {len(chunks)} semantic chunks")
        
        # Create documents with metadata
        documents = self.create_documents_with_metadata(chunks, file_path)
        
        # Create index
        index = self.create_index()
        
        # Process in batches
        batch_size = int(os.getenv("BATCH_SIZE", 50))
        total_uploaded = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            vectors = []
            
            for doc in batch:
                # Generate embedding using OpenAI
                embedding = self.get_embedding(doc['text'])
                
                vectors.append({
                    "id": doc['id'],
                    "values": embedding,
                    "metadata": {
                        "text": doc['text'],
                        "source": doc['metadata']['source'],
                        "chunk_index": doc['metadata']['chunk_index'],
                        "chunk_size": doc['metadata']['chunk_size']
                    }
                })
            
            # Upload to Pinecone
            index.upsert(vectors=vectors)
            total_uploaded += len(vectors)
            print(f"Uploaded batch {i//batch_size + 1} ({len(vectors)} vectors)")
        
        print(f"Story ingestion completed successfully!")
        print(f"Total chunks uploaded: {total_uploaded}")
        
        # Verify the upload
        stats = index.describe_index_stats()
        print(f"Index statistics: {stats}")

if __name__ == "__main__":
    ingestor = StoryIngestor()
    ingestor.ingest_story("story.txt")