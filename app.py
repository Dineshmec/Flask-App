from flask import Flask, render_template, request, jsonify
import os
from flask_cors import CORS
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import random
 
app = Flask(__name__)
CORS(app, resources={r"/query": {"origins": "http://10.10.10.121:8081"}})
 
# Set up Hugging Face token
HF_token = "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"
os.environ['HuGGINGFACEHUB_API_TOKEN'] = HF_token
 
URL = "https://docs.google.com/spreadsheets/d/1CCiYSa2ZReP4gcAIbKflING7eVmwEXAx/edit?usp=sharing&ouid=114895549194268657828&rtpof=true&sd=true"
data = WebBaseLoader(URL)
content = data.load()

# Split the content into chunks
chunk_size = random.randint(4000, 6000)
text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=0)
chunking = text_splitter.split_documents(content)
 
# Get embeddings
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_token,
    model_name="pinecone/bert-retriever-squad2"
)
 
# Create a vector store
vectorstore = Chroma.from_documents(chunking, embeddings)
 
# Set up the Hugging Face model
model = HuggingFaceHub(repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                        model_kwargs={"temperature": 0.5, "max_new_tokens": 512, "max_length": 64},
                        huggingfacehub_api_token="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")

key_size = random.randint(0, 5)

 
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": key_size})
 
qa = RetrievalQA.from_chain_type(llm=model, retriever=retriever, chain_type="stuff")
 
 
@app.route('/query', methods=['POST'])
def process_query():
    try:
        print("Request received")
        # Get the query from the request data
        data = request.get_json()
        query = data.get('query')
        print(query)
        # query = f" act as a bank employee , Write an email to your customer {customer_name} with details in vector DB"
        # prompt = f"""
        # </s>
        # {query}
        # </s>
        # """
        # Use the loaded model to generate a response
        response = qa.invoke(query)
        output = response['result']
        start_index = output.find("Helpful Answer:") + len("Helpful Answer:")
        helpful_answer = output[start_index:].strip()
 
        # Constructing the email format
        #email_subject = f"Transaction Details for {customer_name}"
        #email_content = f"Subject: {email_subject}\n\nDear {customer_name},\n\n{helpful_answer}\n\nBest regards,\n[Your Name]"
 
        # Return the email content
        #return email_content
        return helpful_answer
        #return jsonify({'helpful_answer': helpful_answer})
 
    except Exception as e:
        # Handle exceptions gracefully
        return str(e), 500