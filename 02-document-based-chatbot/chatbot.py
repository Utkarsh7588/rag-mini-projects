import os
import gradio as gr
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

class StoryChatbot:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.index_name = os.getenv("PINECONE_INDEX_NAME")
        self.top_k = int(os.getenv("TOP_K_RESULTS", 5))

        # Initialize Pinecone client (v5+)
        self.pc = Pinecone(api_key=self.pinecone_api_key)

        # Connect to or create the index
        try:
            if self.index_name not in self.pc.list_indexes().names():
                print(f"Creating Pinecone index: {self.index_name}...")
                self.pc.create_index(
                    name=self.index_name,
                    dimension=1536,  # embedding size
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=os.getenv("PINECONE_CLOUD"),
                        region=os.getenv("PINECONE_REGION")
                    )
                )
                # Wait for the index to be ready
                time.sleep(60)

            self.index = self.pc.Index(self.index_name)
            self.index.describe_index_stats()
            print("Successfully connected to Pinecone index")
        except Exception as e:
            print(f"Error connecting to Pinecone: {e}")
            raise

    def get_embedding(self, text):
        """Get embedding for text using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                input=text,
                model="text-embedding-ada-002"
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            raise

    def retrieve_relevant_chunks(self, query):
        """Retrieve relevant story chunks from Pinecone"""
        try:
            query_embedding = self.get_embedding(query)
            results = self.index.query(
                vector=query_embedding,
                top_k=self.top_k,
                include_metadata=True,
                include_values=False
            )
            relevant_chunks = []
            for match in results.matches:
                if match.metadata and 'text' in match.metadata:
                    relevant_chunks.append({
                        'text': match.metadata['text'],
                        'score': match.score,
                        'source': match.metadata.get('source', 'unknown'),
                        'chunk_index': match.metadata.get('chunk_index', 0)
                    })
            relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
            return relevant_chunks
        except Exception as e:
            print(f"Error retrieving chunks: {e}")
            return []

    def generate_answer(self, question, context_chunks):
        """Generate answer using OpenAI based on context"""
        if not context_chunks:
            return "I don't have enough information from the story to answer this question."

        context_texts = []
        for i, chunk in enumerate(context_chunks):
            context_texts.append(f"[Source: {chunk['source']}, Chunk: {chunk['chunk_index']}]")
            context_texts.append(chunk['text'])
            context_texts.append("")

        context = "\n".join(context_texts)
        prompt = f"""# ROLE: Story Knowledge Assistant
You are an expert assistant that answers questions based ONLY on the provided story context.

# STORY CONTEXT:
{context}

# QUESTION:
{question}

# INSTRUCTIONS:
1. Answer ONLY using the story context.
2. If not found, respond: "I cannot answer that based on the story content available."
3. Keep answers concise and accurate.

# ANSWER:"""

        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions strictly based on the story content."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=600
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"

    def answer_question(self, question):
        """Answer a question based on the story"""
        if not question or not question.strip():
            return "Please ask a question about the story content."
        time.sleep(0.5)
        try:
            relevant_chunks = self.retrieve_relevant_chunks(question)
            return self.generate_answer(question, relevant_chunks)
        except Exception as e:
            return f"Error processing question: {str(e)}"
chatbot = StoryChatbot()

def chat_interface(message, history):
    """Gradio chat interface function with Pinecone debug info"""
    if history is None:
        history = []

    # Retrieve Pinecone chunks
    relevant_chunks = chatbot.retrieve_relevant_chunks(message)
    pinecone_debug = "\n".join(
        [f"{i+1}. [Source: {c['source']}, Chunk: {c['chunk_index']}] Score: {c['score']}\n{c['text']}" 
         for i, c in enumerate(relevant_chunks)]
    ) or "No relevant chunks found."

    # Generate answer
    response = chatbot.generate_answer(message, relevant_chunks)

    # Combine final response with debug info
    response_with_debug = f"{response}\n\n---\n**Pinecone Debug Info:**\n{pinecone_debug}"

    history.append([message, response_with_debug])
    return history


def create_gradio_interface():
    """Create enhanced Gradio web interface"""
    with gr.Blocks(
        title="Story Chatbot - OpenAI Embeddings",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 900px;
            margin: 0 auto;
        }
        .chatbot {
            min-height: 500px;
            border: 1px solid #e0e0e0;
            border-radius: 10px;
        }
        .header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """
    ) as demo:
        
        # Header
        with gr.Column(elem_classes="header"):
            gr.Markdown(
                """
                # 📖 Story Chatbot with OpenAI Embeddings
                ### Ask questions about your uploaded story!
                """
            )
        
        # Main chat area
        with gr.Row():
            with gr.Column(scale=3):
                chatbot_interface = gr.Chatbot(
                    label="Story Conversation",
                    height=500,
                    show_copy_button=True,
                    show_label=True,
                    bubble_full_width=False
                )
                
                with gr.Row():
                    message_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Type your question about the story here...",
                        lines=2,
                        max_lines=4,
                        scale=4
                    )
                    submit_btn = gr.Button("🚀 Send", variant="primary", scale=1)
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")
            
            # Sidebar with examples and info
            with gr.Column(scale=1):
                gr.Markdown("### 💡 Example Questions")
                examples = gr.Examples(
                    examples=[
                        "Who is the main character and what are they like?",
                        "What is the central conflict or challenge in the story?",
                        "Describe the setting and atmosphere of the story",
                        "What important events or discoveries happen?",
                        "How does the story end or what is the resolution?"
                    ],
                    inputs=message_input,
                    label="Try these:"
                )
                
                gr.Markdown("### ℹ️ About")
                gr.Markdown("""
                - Uses OpenAI embeddings for semantic search
                - Powered by Pinecone vector database
                - Answers based ONLY on your story content
                - No external knowledge used
                """)
        
        # Event handlers
        submit_btn.click(
            fn=chat_interface,
            inputs=[message_input, chatbot_interface],
            outputs=[chatbot_interface],
            queue=True
        ).then(
            lambda: "", None, message_input
        )
        
        message_input.submit(
            fn=chat_interface,
            inputs=[message_input, chatbot_interface],
            outputs=[chatbot_interface],
            queue=True
        ).then(
            lambda: "", None, message_input
        )
        
        clear_btn.click(
            lambda: None, None, chatbot_interface, queue=False
        )
    
    return demo

if __name__ == "__main__":
    # Create and launch the Gradio interface
    print("Starting Story Chatbot Web Application...")
    print("Server will be available at: http://localhost:7860")
    
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )