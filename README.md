# StudyBuddy AI - RAG Chatbot 📚

A Retrieval-Augmented Generation (RAG) version of StudyBuddy AI, built with Python, Streamlit, ChromaDB and the Groq API.
It answers study questions **only from the documents in the `docs/` folder** and shows the sources used.

## How RAG works here
1. Load PDF/TXT files from `docs/`
2. Clean the text and split it into small chunks
3. Convert chunks to embeddings and store them in ChromaDB
4. Find the chunks most similar to the student's question
5. Send the question and chunks to Groq to write the answer
6. If nothing relevant is found: "I don't know about this information."

## Features
- Answers based on your own study notes (PDF and TXT)
- Source file and page shown for every answer
- Upload new documents from the sidebar
- Follow-up question support (conversation memory)
- Prompt-injection protection
- Clear conversation button

## Documents (docs/ folder)
- `StudyBuddy_Study_Notes.pdf` - 17 pages: Python, Databases/SQL, Operating Systems, Algorithms, Calculus, Statistics, Study Skills
- `StudyBuddy_Guide.pdf` - purpose, allowed topics and answering rules
- `StudyBuddy_README.pdf` / `.txt` - project README

To add more subjects later, just put new PDF/TXT files in `docs/` (or upload them from the sidebar). The knowledge base is rebuilt automatically.

## Installation
1. `pip install -r requirements.txt`
2. Copy `.env.example` to `.env` and add your Groq API key
3. Put your study PDF/TXT files in `docs/`
4. Run: `python -m streamlit run app.py`

The first run downloads a small embedding model (about 80 MB), so internet is needed.

## Security
Never upload your `.env` file or expose your API key.
