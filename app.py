import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util
import os
import zipfile

load_dotenv()

# ── Hallucination verifier ──────────────────────────────────────────────────
verifier_model = SentenceTransformer('all-MiniLM-L6-v2')

def check_hallucination(answer, retrieved_chunks, threshold=0.35):
    sentences = [s.strip() for s in answer.split('.') if len(s.strip()) > 20]
    chunk_texts = [doc.page_content for doc in retrieved_chunks]
    results = []
    for sentence in sentences:
        sent_emb = verifier_model.encode(sentence, convert_to_tensor=True)
        chunk_embs = verifier_model.encode(chunk_texts, convert_to_tensor=True)
        scores = util.cos_sim(sent_emb, chunk_embs)[0]
        max_score = float(scores.max())
        results.append({
            "sentence": sentence,
            "confidence": round(max_score, 2),
            "hallucinated": max_score < threshold
        })
    return results

# ── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Your Documents")
    files_sidebar = st.file_uploader("Upload your files", accept_multiple_files=True, key="sidebar_uploader")

# ── Header ───────────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    st.markdown("<h1 style='text-align: center; color: white;'>Hello Coders!!</h1>", unsafe_allow_html=True)

# ── Main uploader ────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    with st.expander("Upload Your code files", expanded=True):
        files_main = st.file_uploader("Upload your files", accept_multiple_files=True, key="main_uploader")

# ── Combine files ────────────────────────────────────────────────────────────
files = list(set(list(files_sidebar or []) + list(files_main or [])))

# ── Chat memory ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Document processing ───────────────────────────────────────────────────────
documents = []

if files:
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n"],
        chunk_size=1000,
        chunk_overlap=150
    )

    for file in files:

        # ZIP
        if file.name.endswith('.zip'):
            with zipfile.ZipFile(file, 'r') as zip_ref:
                file_list = "\n".join(zip_ref.namelist())
                documents.append(Document(
                    page_content=f"Files inside zip:\n{file_list}",
                    metadata={"source_file": file.name, "internal_file": "ZIP_MANIFEST",
                              "page": "Index", "chunk": 0, "type": "zip_manifest", "file_path": file.name}
                ))
                for info in zip_ref.infolist():
                    if info.filename.endswith(('.py', '.js', '.ts', '.java', '.cpp')):
                        try:
                            content = zip_ref.read(info).decode('utf-8', errors='ignore')
                        except:
                            continue
                        chunks = text_splitter.split_text(content)
                        for chunk_idx, chunk in enumerate(chunks, 1):
                            documents.append(Document(
                                page_content=chunk,
                                metadata={"source_file": file.name, "internal_file": info.filename,
                                          "page": f"Chunk {chunk_idx}", "chunk": chunk_idx,
                                          "type": "code", "file_path": f"{file.name} → {info.filename}",
                                          "total_chunks": len(chunks)}
                            ))

        # PDF
        elif file.name.endswith('.pdf'):
            pdf_reader = PdfReader(file)
            for i, page in enumerate(pdf_reader.pages):
                content = page.extract_text() or ""
                chunks = text_splitter.split_text(content)
                for chunk_idx, chunk in enumerate(chunks, 1):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={"source_file": file.name, "internal_file": None,
                                  "page": i + 1, "chunk": chunk_idx, "type": "pdf",
                                  "file_path": file.name, "total_chunks": len(chunks)}
                    ))

        # Text / code files
        else:
            try:
                content = file.read().decode('utf-8', errors='ignore')
            except:
                continue
            chunks = text_splitter.split_text(content)
            for chunk_idx, chunk in enumerate(chunks, 1):
                documents.append(Document(
                    page_content=chunk,
                    metadata={"source_file": file.name, "internal_file": None,
                              "page": f"Chunk {chunk_idx}", "chunk": chunk_idx,
                              "type": "text", "file_path": file.name, "total_chunks": len(chunks)}
                ))

    # ── Embeddings & vector store ─────────────────────────────────────────────
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(documents, embedding=embeddings)

    llm = ChatGroq(model_name="llama-3.1-8b-instant", temperature=0)

    user_question = st.chat_input("Ask something...")

    if user_question:
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.write(user_question)

        docs = vector_store.similarity_search(user_question, k=5)
        context = "\n\n".join([doc.page_content for doc in docs])

        prompt = f"""
You are a senior software engineer and expert coding assistant.

Step 1: Directly answer the user's question clearly.
Step 2: Then analyze the provided context for deeper insights.

Instructions:
- Answer FIRST in 2-4 lines
- Then give detailed explanation
- If code is present, explain it
- If bug exists, identify and fix it
- Mention file names where relevant
- If question is about files, list them clearly
- If not found, say: "Not found in document"

Context:
{context}

Question:
{user_question}

Answer:
"""

        with st.spinner("Thinking..."):
            response = llm.invoke(prompt)

        # ── Answer ────────────────────────────────────────────────────────────
        with st.chat_message("assistant"):
            st.write(response.content)

        # ── Hallucination Analysis ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 🔬 Hallucination Analysis")

        hal_results = check_hallucination(response.content, docs)
        hallucinated = [r for r in hal_results if r["hallucinated"]]
        safe = [r for r in hal_results if not r["hallucinated"]]
        total = len(hal_results)
        hal_rate = round(len(hallucinated) / total * 100, 1) if total > 0 else 0

        col1, col2, col3 = st.columns(3)
        col1.metric("✅ Grounded", len(safe))
        col2.metric("⚠️ Hallucinated", len(hallucinated))
        col3.metric("🎯 Confidence", f"{100 - hal_rate}%")

        for r in hal_results:
            color = "🔴" if r["hallucinated"] else "🟢"
            st.markdown(f"{color} *{r['sentence']}* — score: `{r['confidence']}`")

        # ── Sources ───────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### 📚 Sources Used:")

        sources_by_file = {}
        for doc in docs:
            source_file = doc.metadata.get("source_file", "Unknown")
            if source_file not in sources_by_file:
                sources_by_file[source_file] = []
            sources_by_file[source_file].append(doc.metadata)

        for source_file, meta_list in sources_by_file.items():
            file_icon = "📦" if source_file.endswith('.zip') else "📄"
            st.markdown(f"**{file_icon} {source_file}**")
            for meta in meta_list:
                internal_file = meta.get("internal_file")
                page = meta.get("page")
                doc_type = meta.get("type", "").upper()
                file_path = meta.get("file_path", source_file)
                if internal_file and internal_file != "ZIP_MANIFEST":
                    st.write(f"  └─ **{internal_file}** ({page}) `{doc_type}`")
                elif internal_file == "ZIP_MANIFEST":
                    st.write(f"  └─ **ZIP Contents Index** `MANIFEST`")
                else:
                    st.write(f"  └─ **{file_path}** ({page}) `{doc_type}`")

        st.markdown(f"**Total chunks analyzed:** {len(docs)}")

        st.session_state.messages.append({"role": "assistant", "content": response.content})
