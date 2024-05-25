from flask import Flask, render_template, request, jsonify
import os
from flask_cors import CORS
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
import random
from huggingface_hub import login

app = Flask(__name__)
CORS(app, resources={r"/query": {"origins": "http://10.10.10.121:8081"}})

# Set up Hugging Face API token
HF_token = "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"
login(token=HF_token, add_to_git_credential=True)  # Logging into Hugging Face Hub and saving to git credentials

# Load data from the provided URL
URL = "https://docs.google.com/spreadsheets/d/1EaMOBcQgADrHbFgR1qwUFfC9ZlPiLOgb/edit?usp=sharing&ouid=109582196542583521183&rtpof=true&sd=true"
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
model = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                            temperature=0.5, 
                            max_new_tokens=512, 
                            max_length=64,
                            huggingfacehub_api_token=HF_token)

# model = HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2",
#                         model_kwargs={"temperature": 0.5, "max_new_tokens": 512, "max_length": 64},
#                         huggingfacehub_api_token="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")

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
        # Use the loaded model to generate a response
        response = qa.invoke("Act as a Mahendra engineering college advisor and provide me the answer only for what i am asking ,"+query)
        output = response['result']
        start_index = output.find("Helpful Answer:") + len("Helpful Answer:")
        helpful_answer = output[start_index:].strip()

        # Return the helpful answer
        return output

    except Exception as e:
        # Handle exceptions gracefully
        return str(e), 500
