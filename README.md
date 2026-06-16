# Rag-Chatbot
Link for the chatbot:
https://huggingface.co/spaces/vanshhh-18/RAG-Chatbot

# Why This Chatbot is Useful for Code Files

## Quick Overview

This chatbot is great for understanding and working with code. Upload your code files (Python, JavaScript, etc.) and ask questions about them instead of manually searching through files.

It uses RAG (Retrieval-Augmented Generation) to search through your code and find relevant answers.

## Use Cases

**Understanding Code**
- Upload a project and ask "What does this file do?"
- Get explanations of specific functions or classes
- Understand how different files work together

**Debugging**
- Ask "Why is this function failing?"
- Get help finding issues in your code
- Understand error handling

**Onboarding**
- New developers can upload the project and ask questions
- No need to wait for senior devs to explain things
- Self-serve learning

**Code Review**
- Ask about specific parts before reviewing
- Understand the architecture and flow
- Check dependencies and tech stack

## How It Works

1. Upload your code files (or ZIP archive)
2. Ask your question in natural language
3. Get answers with sources showing which file was used
4. See the exact chunk/line that was referenced

## RAG Flow

```
Upload Code → Split into Chunks → Convert to Embeddings → Store in Vector DB
                                                              ↓
                                              User Question → Find Similar Chunks
                                                              ↓
                                              LLM Reads Chunks → Answer with Sources
```

In simple terms:
1. Your code is converted to vectors (embeddings)
2. When you ask a question, it's also converted to a vector
3. The system finds the most similar code chunks
4. The LLM reads those chunks and answers your question

## Supported Languages

- Python (.py)
- JavaScript (.js)
- TypeScript (.ts)
- Java (.java)
- C++ (.cpp)

## Example

```
You: "Explain the error handling in this project"

Bot: "Your code uses try-except blocks in main.py 
to handle file reading errors. The error messages 
are logged and displayed to the user..."

📚 Sources Used:
📄 main.py (Chunk 2) [CODE]
```

You know exactly which file and part to check!

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Groq (Llama 3.1 8B) - Fast inference
- **Embeddings**: HuggingFace (all-MiniLM-L6-v2) - Convert code to vectors
- **Vector DB**: FAISS - Store and search embeddings
- **Text Processing**: LangChain - Split and manage text
- **PDF**: PyPDF - Extract text from PDFs


