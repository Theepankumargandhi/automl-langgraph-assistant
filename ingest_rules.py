# ingest_rules.py
import os
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
try:
    # Newer LangChain
    from langchain_core.documents import Document  # type: ignore
except Exception:
    # Back-compat
    from langchain.schema import Document  # type: ignore

load_dotenv()

# ----------------------
# Few-shot patterns (optional to ingest)
# ----------------------
FEWSHOT_CLASSIFICATION = """
# Pattern: Robust classification pipeline (fits Code Contract)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression

X = df.drop(columns=[target_column])
y = df[target_column]

num_cols = X.select_dtypes(include=['number','bool']).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

pre = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
    ]
)

model = Pipeline(steps=[('pre', pre), ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))])
_ = model.predict(X.head(1))
"""

FEWSHOT_REGRESSION = """
# Pattern: Robust regression pipeline (fits Code Contract)
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

X = df.drop(columns=[target_column])
y = df[target_column]

num_cols = X.select_dtypes(include=['number','bool']).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

pre = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='median'), num_cols),
        ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')),
                                ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols),
    ]
)

model = Pipeline(steps=[('pre', pre), ('clf', RandomForestRegressor(n_estimators=300, random_state=42))])
_ = model.predict(X.head(1))
"""

def fewshot_snippets(profile: dict) -> str:
    """Return a task-appropriate few-shot snippet based on the dataset profile."""
    t = (profile or {}).get("target_type", "classification").lower()
    return FEWSHOT_REGRESSION if "regress" in t else FEWSHOT_CLASSIFICATION


# ----------------------
# Ingestion helpers
# ----------------------
def _require_env(key: str) -> None:
    if not os.getenv(key):
        raise EnvironmentError(
            f"Missing environment variable {key!r}. "
            f"Add it to your .env or environment before running ingestion."
        )

def _read_rule_docs(rules_dir: str = "rules") -> List[Document]:
    """
    Read *.md files from rules_dir, skipping hidden/empty files.
    Uses filename (without path) as a stable id so re-running updates content instead of duplicating.
    """
    p = Path(rules_dir)
    if not p.is_dir():
        raise FileNotFoundError(f"Rules directory not found: {p.resolve()}")

    docs: List[Document] = []
    for fp in sorted(p.glob("*.md")):
        if fp.name.startswith("."):
            continue
        text = fp.read_text(encoding="utf-8").strip()
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={"source": fp.name, "kind": "rule"},
            )
        )
    if not docs:
        raise ValueError(f"No valid .md rule files found in {p.resolve()}. Add markdown rule documents first.")
    return docs

def _fewshot_docs() -> List[Document]:
    return [
        Document(page_content=FEWSHOT_CLASSIFICATION.strip(), metadata={"source": "fewshot_classification.py", "kind": "fewshot"}),
        Document(page_content=FEWSHOT_REGRESSION.strip(),     metadata={"source": "fewshot_regression.py",     "kind": "fewshot"}),
    ]


def ingest_rules(
    rules_dir: str = "rules",
    persist_dir: str = "chroma_store/rules",
    collection_name: str = "rules",
    include_fewshots: bool = True,
) -> int:
    """
    One-time (or repeatable) ingestion of markdown rules (and optional few-shot patterns)
    into a Chroma vector store. Returns the number of documents ingested/updated.
    Re-running is idempotent at the collection level.
    """
    _require_env("OPENAI_API_KEY")
    Path(persist_dir).mkdir(parents=True, exist_ok=True)

    docs = _read_rule_docs(rules_dir)
    if include_fewshots:
        docs.extend(_fewshot_docs())

    print(f"Loaded {len(docs)} documents (rules_dir='{rules_dir}', include_fewshots={include_fewshots}).")

    # Use stable ids derived from metadata['source'] to upsert rather than duplicate.
    ids = [d.metadata.get("source", f"doc_{i}") for i, d in enumerate(docs)]

    _ = Chroma.from_documents(
        documents=docs,
        ids=ids,
        embedding=OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY")),
        collection_name=collection_name,
        persist_directory=persist_dir,
    )

    print(f"Ingested/updated {len(docs)} documents to collection '{collection_name}' at '{persist_dir}'.")
    return len(docs)


if __name__ == "__main__":
    # Allow overrides via env vars
    RULES_DIR   = os.getenv("RULES_DIR", "rules")
    PERSIST_DIR = os.getenv("RULES_PERSIST_DIR", "chroma_store/rules")
    COLLECTION  = os.getenv("RULES_COLLECTION", "rules")
    FEWSHOTS    = os.getenv("RULES_INCLUDE_FEWSHOTS", "1") not in {"0", "false", "False"}

    ingest_rules(RULES_DIR, PERSIST_DIR, COLLECTION, include_fewshots=FEWSHOTS)
