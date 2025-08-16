# retrieval_agent.py
from typing import List, Tuple, Optional
import os
import json
from dotenv import load_dotenv

# Few-shot blueprint appended to whatever we retrieve (stabilizes LLM codegen)
from ingest_rules import fewshot_snippets

load_dotenv()


def _append_fewshot(profile: dict, rules: List[str]) -> None:
    """Always add a compact few-shot snippet based on the current task."""
    try:
        snippet = fewshot_snippets(profile)
        if snippet and snippet.strip():
            rules.append(snippet.strip())
    except Exception:
        # Non-fatal: we still return whatever we have
        pass


def retrieve_rules(
    profile: dict,
    chroma_client: Optional[object] = None,
    max_distance: float = 0.30,
    k: int = 5,
) -> Tuple[List[str], bool]:
    """
    Retrieve relevant rules from Chroma. Chroma returns *distance* (lower is better).
    Keep docs where distance <= max_distance. If the vectorstore is unavailable or
    nothing passes the threshold, we still append a task-appropriate few-shot code
    pattern so the LLM has a solid example to imitate.

    Returns
    -------
    (rules, fallback_mode)
      rules         : list of rule strings (few-shot snippet is always included)
      fallback_mode : True if no vectorstore rules matched (distance > threshold or error)
    """
    from langchain_openai import OpenAIEmbeddings
    from langchain_chroma import Chroma

    profile_text = json.dumps(profile, indent=2, default=str)

    # Try using provided client first; if that fails, fall back to local persisted store.
    retrieved_rules: List[str] = []
    fallback_mode = True  # assume fallback until we confirm a match

    try:
        embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

        # Attempt with explicit client if provided (e.g., server mode)
        vectorstore = None
        if chroma_client is not None:
            try:
                vectorstore = Chroma(
                    client=chroma_client,
                    collection_name="rules",
                    embedding_function=embeddings,
                    persist_directory="chroma_store/rules",
                )
            except Exception:
                vectorstore = None  # will try local init next

        # If no client init, use local persisted collection (in-process)
        if vectorstore is None:
            vectorstore = Chroma(
                collection_name="rules",
                embedding_function=embeddings,
                persist_directory="chroma_store/rules",
            )

        results = vectorstore.similarity_search_with_score(profile_text, k=k)

        # Filter by distance threshold (smaller is more similar)
        for doc, dist in results:
            try:
                if dist <= max_distance:
                    retrieved_rules.append(doc.page_content)
            except Exception:
                # If dist isn’t a number for some reason, skip it
                continue

        fallback_mode = len(retrieved_rules) == 0

    except Exception:
        # Vector store missing / embeddings not configured / any other retrieval issue
        retrieved_rules = []
        fallback_mode = True

    # Always append a compact few-shot pattern grounded in the current task
    _append_fewshot(profile, retrieved_rules)

    return retrieved_rules, fallback_mode
