"""
llm.py
Gemini API wrapper for answer generation from retrieved context.
"""

import os
import logging
from typing import Optional

import google.generativeai as genai

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise question-answering assistant for Swiggy's Annual Report FY 2023-24.

Rules you must follow without exception:
1. Answer ONLY using the provided context passages. Do not use any external knowledge.
2. If the answer is not found in the context, respond exactly: "The provided document does not contain enough information to answer this question."
3. Be factual, concise, and cite the page number(s) when mentioning specific figures or statements.
4. Do not speculate, extrapolate, or add commentary beyond what the context states.
5. Format numbers and financial figures exactly as they appear in the source.
"""


class GeminiLLM:
    """Thin wrapper around the Gemini generative API."""

    def __init__(self, api_key: Optional[str] = None, model: str = "gemini-2.5-flash"):
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY is not set. "
                "Add it to your .env file or export it as an environment variable."
            )
        genai.configure(api_key=key)
        self.model_name = model
        self.client = genai.GenerativeModel(
            model_name=model,
            system_instruction=SYSTEM_PROMPT,
        )
        logger.info("GeminiLLM initialized with model: %s", model)

    def answer(self, question: str, context_chunks: list[dict]) -> dict:
        """
        Generate an answer given a question and a list of retrieved context chunks.

        Returns:
            {
                "answer": str,
                "context_used": list[dict]  # the chunks passed in, with page info
            }
        """
        if not context_chunks:
            return {
                "answer": "No relevant context was found in the document for your query.",
                "context_used": [],
            }

        # Build the context block
        context_lines = []
        for i, chunk in enumerate(context_chunks, start=1):
            context_lines.append(
                f"[Context {i} | Page {chunk['page']}]\n{chunk['text']}"
            )
        context_block = "\n\n---\n\n".join(context_lines)

        prompt = (
            f"Using ONLY the context passages below, answer the following question.\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{context_block}"
        )

        logger.info("Sending prompt to Gemini (%d context chunks)", len(context_chunks))
        response = self.client.generate_content(prompt)
        answer_text = response.text.strip()

        return {
            "answer": answer_text,
            "context_used": context_chunks,
        }