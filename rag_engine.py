"""
rag_engine.py
-------------
Implements the "RAG" (Retrieval-Augmented Generation) capability.

Documents (FAQs, syllabus, regulations, notices) are loaded from the
data/ folder, split into chunks, and indexed with TF-IDF vectors. At query
time, the top-matching chunks are retrieved by cosine similarity and used
as grounding context for the answer.

This is deliberately implemented with scikit-learn's TF-IDF (instead of a
paid embeddings API) so the whole project runs fully offline with no API
keys required. If an OPENAI_API_KEY (or similar) is set in the environment,
generate_answer() will use it to phrase a more natural answer on top of the
retrieved context; otherwise it falls back to a clean extractive answer.
"""

import os
import glob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RAGEngine:
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.chunks = []       # list of dicts: {text, source}
        self.vectorizer = None
        self.doc_matrix = None
        self._build_index()

    def _load_documents(self):
        docs = []
        for filepath in glob.glob(os.path.join(self.data_dir, "*.txt")):
            source = os.path.basename(filepath)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            # Chunk on blank-line-separated blocks (each FAQ / course / regulation)
            blocks = [b.strip() for b in content.split("\n\n") if b.strip()]
            for block in blocks:
                docs.append({"text": block, "source": source})
        return docs

    def _build_index(self):
        self.chunks = self._load_documents()
        if not self.chunks:
            self.vectorizer = None
            self.doc_matrix = None
            return
        texts = [c["text"] for c in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(texts)

    def reload(self):
        """Call after adding new documents to data/ to refresh the index."""
        self._build_index()

    def retrieve(self, query: str, top_k: int = 3):
        if not self.chunks or self.vectorizer is None:
            return []
        query_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(query_vec, self.doc_matrix).flatten()
        ranked_idx = sims.argsort()[::-1][:top_k]
        results = []
        for idx in ranked_idx:
            if sims[idx] > 0.05:  # ignore near-zero matches
                results.append(
                    {
                        "text": self.chunks[idx]["text"],
                        "source": self.chunks[idx]["source"],
                        "score": float(sims[idx]),
                    }
                )
        return results

    def generate_answer(self, query: str, top_k: int = 3):
        """
        Returns (answer_text, retrieved_chunks).
        Uses an LLM to phrase the answer if OPENAI_API_KEY is set, otherwise
        returns a clean extractive answer built directly from the top chunk.
        """
        retrieved = self.retrieve(query, top_k=top_k)
        if not retrieved:
            return (
                "I couldn't find anything about that in the college's FAQs, "
                "syllabus, or regulations. Could you rephrase, or would you "
                "like me to escalate this to the Admin Office?",
                [],
            )

        api_key = os.environ.get("OPENAI_API_KEY")
        if api_key:
            try:
                return self._generate_with_llm(query, retrieved), retrieved
            except Exception:
                pass  # fall back silently to extractive mode

        # Extractive fallback: return the best-matching chunk directly.
        best = retrieved[0]
        answer = best["text"]
        return answer, retrieved

    def _generate_with_llm(self, query: str, retrieved: list) -> str:
        from openai import OpenAI

        client = OpenAI()
        context = "\n\n".join(f"[{r['source']}] {r['text']}" for r in retrieved)
        prompt = (
            "You are a helpful college support assistant. Answer the "
            "student's question using ONLY the context below. If the "
            "context does not contain the answer, say you don't know and "
            "suggest contacting the relevant office.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
