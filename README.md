# **LegalRAG**  

A Retrieval-Augmented Generation (RAG) system for legal document analysis, built with **FastAPI**, **Pinecone**, and **Llama 3.3 via Groq Inference**.  

## **Overview**  

LegalRAG is an AI-powered system that retrieves and analyzes legal documents based on user queries. It utilizes **vector-based search** to provide relevant case law, legal statutes, and precedents in response to legal queries.  

## **Features**  

**Legal Document Retrieval** – Uses Pinecone for fast and efficient case law retrieval.  
 **Natural Language Querying** – Allows users to search legal documents using plain English.  
 **Session-Based Handling** – Supports session-based query tracking for better user experience.  
 **FastAPI Backend** – Provides a scalable and efficient REST API.  
 **Concurrency Handling** – Manages multiple legal research sessions efficiently.  
 **Llama 3.3 via Groq Inference** – High-speed, low-latency language model inference.  

---  

## **Installation**  

### **1. Clone the repository**  
```bash
git clone https://github.com/arulkumarann/legalRAG.git
cd legalRAG
```

### **2. Create a virtual environment**  
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### **3. Install dependencies**  
```bash
pip install -r requirements.txt
```

---  

## **Configuration**  

Create a `.env` file and add the following variables:  
```
PINECONE_API_KEY=your_pinecone_api_key
INDEX_NAME=your_pinecone_index_name
GROQ_API_KEY=your_groq_api_key
```

---  

## **Running the Application**  

### **Local Development**  
```bash
uvicorn app.main:app --reload
```
API will be available at **http://localhost:8000**  


---  

## **API Endpoints**  

### **1. Root Check**  
- **GET `/`** – Health check endpoint  

### **2. Chat with RAG**  
- **POST `/chat`** – Query legal documents with RAG  
  - **Body:**  
    ```json
    {
      "session_id": "abc123",
      "query": "What are the key points of Smith v. Johnson?"
    }
    ```
  - **Response:**  
    ```json
    {
      "response": "The key points are breach of contract and negligence...",
      "sources": ["case_smith_v_johnson.pdf"]
    }
    ```

### **3. Session Management**  
- **DELETE `/session/{session_id}`** – End a legal research session  
- **POST `/session/reset/{session_id}`** – Reset a session's state  
- **GET `/sessions/count`** – Get the total number of active sessions  

### **4. Retrieve Source Documents**  
- **GET `/sources/{session_id}`** – Fetch relevant legal documents for a session  

---  

## **Llama 3.3 via Groq Inference**  

- **Inference Speed:** <15ms per query response  
- **Token Throughput:** ~500 tokens/sec  
- **Latency:** Low-latency response optimized for real-time queries  
- **Memory Usage:** Efficient memory footprint compared to traditional on-device LLMs  
- **Scalability:** Supports concurrent user queries without degradation  

---  

## **Concurrency & Session Handling**  

This RAG system supports multiple concurrent users by assigning unique `session_id` values for each session.  

- **Sessions track user queries** to improve context and accuracy.  
- **Each session has its own vector search scope** in Pinecone, ensuring faster retrieval of case-specific documents.  
- **Automatic cleanup** – Sessions can be ended manually (`DELETE /session/{session_id}`) or reset (`POST /session/reset/{session_id}`).  

---  

## **Deployment**  

### **Deploy on Render**  
1. **Build Command:**  
   ```bash
   pip install -r requirements.txt
   ```
2. **Start Command:**  
   ```bash
   gunicorn app.main:app -k uvicorn.workers.UvicornWorker --workers 1 --threads 2 --timeout 120
   ```
3. **Environment Variables:**  
   - `PINECONE_API_KEY`  
   - `INDEX_NAME`  
   - `GROQ_API_KEY`  

---  

## **Memory Optimization**  

- Uses **lazy model loading** to reduce memory usage.  
- **Batch processing** to prevent memory spikes.  
- **Garbage collection** after query execution.  

---  


## **License**  
This project is open-source under the **MIT License**.  
