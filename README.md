# 💬 PDFCrawler (RAG + Ollama + Gradio)

This project is a **Retrieval-Augmented Generation (RAG)** based chatbot that lets users ask questions about their PDF documents. It leverages:

- 🧠 **Ollama LLMs** (e.g., Mistral)
- 📚 **Sentence Transformers** for embeddings
- 🔍 **FAISS** for vector similarity search
- 🖥️ **Gradio** for a web-based chatbot interface

---

## Pipeline 
![alt text](https://miro.medium.com/v2/resize:fit:1400/format:webp/0*KRJtcCOw3buwG6xV.png)

The pipeline involves parsing PDFs, splitting them into chunks, and storing embeddings in a FAISS vector database. The chatbot retrieves relevant document sections using these embeddings and answers user queries through a custom-trained language model (Ollama, e.g., Mistral). A Gradio-based web interface allows users to interact with the chatbot, which provides answers, validation against expected answers, and source information.

---

## 📁 Project Structure
```bash
├── chatbot.py # RAG chatbot logic using Ollama + FAISS
├── vectordb_setup.py # Creates vector DB from PDF files
├── live-chat.py # Gradio web interface
├── files/ # Place your PDF files here
├── query.json # (Optional) Ground-truth Q&A for validation
├── vectordb/ # Saved FAISS vector DB (auto-generated)
├── requirements.txt # Python dependencies
└── setup.sh # Virtual environment + dependency setup
```

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/PDFCrawler.git
cd PDFCrawler
```

### 2. Run the Setup Script
```bash
bash setup.sh
```

This will:
* Create a Python virtual environment
* Install all dependencies listed in requirements.txt
* Install and configure Ollama (if not already installed)

## 📦 How to Use
### Step 1: Add PDFs
    Place all your .pdf files inside the files/ folder.

### Step 2: Create the Vector Database
```bash
python3 vectordb_setup.py
```

This will:
* Load and split PDFs
* Generate embeddings using Sentence Transformers
* Save vectors into a local FAISS vector store under vectordb/

### Step 3: Start the Chat Interface
```bash
python3 live-chat.py
```
This launches a Gradio web app for you to interact with your PDF-aware chatbot.


## 🧪 Optional: Add Expected Answers
If you'd like to validate chatbot responses, you can include a query.json file in this format:

```json
{
  "queries": [
    {
      "question": "What is the capital of France?",
      "answer": "Paris"
    },
    {
      "question": "Who wrote The Odyssey?",
      "answer": "Homer"
    }
  ]
}
```
The chatbot will compare its response to the expected answer and show ✅/❌ match results.



## 🔍 Models Used

| Component    | Model Name                               |
| ------------ | ---------------------------------------- |
| Embeddings   | `sentence-transformers/all-MiniLM-L6-v2` |
| LLM (Ollama) | `mistral`                                |

You can change these in `live-chat.py` or `chatbot.py`

## 🙌 Acknowledgements
    LangChain
    FAISS
    HuggingFace Sentence Transformers
    Ollama
    Gradio

## 🚀 Future Improvements
- Support for multi-modal PDFs (images + text)
- Session memory for follow-up questions
- Dockerized deployment and Hugging Face Spaces version