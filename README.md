# Swiggy Annual Report — RAG Question Answering System

A Retrieval-Augmented Generation (RAG) application that answers natural language questions about Swiggy's Annual Report FY 2023-24.

All answers are grounded strictly in the source document — no hallucination.

---

## Source Document

Swiggy Annual Report FY 2023-24:  
https://drive.google.com/file/d/1pbH6aoafF-nYz0j6Lw_VKSYtig9m-wzv/view?usp=sharing

---
<p align="center">
  <img src="https://github.com/harshogale04/SwiggyAsk-/blob/main/swiggy2.png?raw=true" width="45%" />
  <img src="https://github.com/harshogale04/SwiggyAsk-/blob/main/swiggy1.png?raw=true" width="45%" />
</p>

---
## Tech Stack

- PyMuPDF — PDF text extraction  
- sentence-transformers (all-MiniLM-L6-v2) — text embeddings  
- FAISS — vector similarity search  
- Gemini 1.5 Flash — answer generation  
- FastAPI — backend server  
- Vanilla HTML/CSS/JS — frontend UI  

---

## Project Structure

```
SwiggyAsk-/
├── backend/
│   ├── rag_engine.py       # PDF processing, chunking, embeddings, FAISS
│   ├── llm.py              # Gemini API wrapper
│   └── main.py             # FastAPI app
├── frontend/
│   └── index.html          # Single-page UI
├── data/
│   ├── swiggy_annual_report.pdf   # place PDF here
│   └── faiss_index/               # auto-created after ingest
├── cli.py                  # Terminal interface
├── ingest.py               # One-time PDF ingestion script
├── requirements.txt
├── .env.example
└── README.md
```

---

# Setup and Running

## 1. Create and Activate Virtual Environment

```bash
python -m venv venv
```

### Mac/Linux
```bash
source venv/bin/activate
```

### Windows
```bash
venv\Scripts\activate
```

---

## 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Configure Environment

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key_here
PDF_PATH=./data/swiggy_annual_report.pdf
INDEX_PATH=./data/faiss_index
EMBED_MODEL=all-MiniLM-L6-v2
TOP_K=5
```

Get a free Gemini API key at:  
https://aistudio.google.com/app/apikey

---

## 4. Place the PDF

Download the Swiggy FY 2023-24 Annual Report and save it as:

```
data/swiggy_annual_report.pdf
```

---

## 5. Ingest the PDF (Run Once)

```bash
python ingest.py
```

This builds the FAISS index and saves it to `data/faiss_index/`.  
Only needs to run once.

---

## 6. Start the Server

```bash
cd backend
uvicorn index.main:app --reload --host 0.0.0.0 --port 8000
```

Open in browser:

```
http://localhost:8000
```

---

## OR Use Terminal Interface

```bash
python cli.py
```

---

# How It Works

1. The PDF is extracted page by page and split into overlapping text chunks  
2. Each chunk is converted into a vector using a sentence-transformer model  
3. All vectors are stored in a FAISS index on disk  
4. When a question is asked, it is also converted to a vector  
5. FAISS finds the top 5 most similar chunks from the document  
6. Those chunks are sent to Gemini with the question  
7. Gemini answers strictly from the provided chunks — no hallucination  

---

# API

## POST `/ask`

```json
{
  "question": "What was Swiggy's revenue in FY 2024?"
}
```

## GET `/health`

Returns system status and number of indexed chunks.

---

# Sample Questions

- What was Swiggy's total revenue in FY 2024?
- How many active delivery partners does Swiggy have?
- What are the key risks mentioned in the annual report?
- What is the board composition of Swiggy?
- What is Instamart and how does it contribute to revenue?
- What was the net loss reported in FY 2024?
- How many cities does Swiggy operate in?

---
