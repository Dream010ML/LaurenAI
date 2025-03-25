# Import necessary libraries
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
import torch

# Step 1: Load the Retrieval Model
# Using SentenceTransformer for document embedding and retrieval
retrieval_model = SentenceTransformer('all-MiniLM-L6-v2')  # Pre-trained embedding model

# Step 2: Load the Generative Model
# Using a pre-trained language model for response generation
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
generative_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")

# Step 3: Prepare the Knowledge Base
# Example knowledge base (can be replaced with a larger dataset or database)
knowledge_base = [
    "Retrieval-Augmented Generation (RAG) combines retrieval and generation models.",
    "RAG is useful for knowledge-intensive tasks in NLP.",
    "NLP chatbots use tokenization, stemming, and sentiment analysis to process text.",
    "RAG chatbots are transforming customer support by providing accurate and contextually relevant responses."
]

# Encode the knowledge base using the retrieval model
knowledge_embeddings = retrieval_model.encode(knowledge_base, convert_to_tensor=True)

# Step 4: Define the Chatbot Function
def rag_chatbot(user_query):
    # Encode the user query
    query_embedding = retrieval_model.encode(user_query, convert_to_tensor=True)
    
    # Retrieve the most relevant document
    scores = util.pytorch_cos_sim(query_embedding, knowledge_embeddings)
    top_doc_index = torch.argmax(scores).item()
    retrieved_doc = knowledge_base[top_doc_index]
    
    # Combine the user query and retrieved document for input to the generative model
    input_text = f"User Query: {user_query}\nRetrieved Document: {retrieved_doc}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    
    # Generate a response
    outputs = generative_model.generate(**inputs, max_length=150, num_beams=5, early_stopping=True)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Step 5: Test the Chatbot
if __name__ == "__main__":
    user_query = "What is RAG and how is it used in NLP?"
    response = rag_chatbot(user_query)
    print("Chatbot Response:", response)
