from typing import List, Tuple
import numpy as np

from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI

from .config import (
    SYSTEM_PROMPT,
    EMBEDDING_MODEL_NAME,
    SAMPLE_DOCS,
    OPENAI_MODEL,
)


class MedicalRAGAgent:
    """Simple RAG + multi-turn medical reasoning agent."""

    def __init__(self):
        self.client = OpenAI()
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.docs = SAMPLE_DOCS
        self.index = self._build_index(self.docs)

    def _build_index(self, docs: List[str]):
        embs = self.embed_model.encode(docs)
        dim = embs.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(np.array(embs))
        return index

    def retrieve(self, query: str, k: int = 2) -> List[str]:
        q_emb = self.embed_model.encode([query])
        D, I = self.index.search(np.array(q_emb), k)
        retrieved = [self.docs[idx] for idx in I[0]]
        return retrieved

    def build_messages(
        self,
        user_query: str,
        history: List[Tuple[str, str]]
    ):
        """Convert history + RAG into OpenAI messages format."""
        # history is [(user, assistant), ...]
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        # incorporate conversation history
        for user_msg, bot_msg in history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": bot_msg})

        # RAG
        retrieved_chunks = self.retrieve(user_query, k=2)
        context_text = "\n\n".join(retrieved_chunks)

        rag_instruction = (
            "You have access to the following high-level medical context.
"
            "Use it to guide your reasoning but do not claim it is a diagnosis.

"
            f"{context_text}
"
        )

        messages.append(
            {
                "role": "system",
                "content": rag_instruction,
            }
        )

        messages.append({"role": "user", "content": user_query})
        return messages

    def generate_response(
        self,
        user_query: str,
        history: List[Tuple[str, str]]
    ) -> str:
        messages = self.build_messages(user_query, history)
        completion = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=400,
            temperature=0.4,
        )
        return completion.choices[0].message.content.strip()