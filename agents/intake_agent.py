# add to agents/intake_agent.py
import re, os, zipfile, tempfile, pandas as pd
from urllib.parse import urlparse, parse_qs
from kaggle import api

def parse_kaggle_input(user_text: str, fallback_file: str | None = None):
    """
    Accepts either a full Kaggle URL or a plain 'owner/dataset' slug.
    Returns (slug, csv_name or None).
    """
    txt = (user_text or "").strip()
    if not txt:
        raise ValueError("Provide a Kaggle dataset slug or URL.")
    if txt.startswith(("http://", "https://")):
        u = urlparse(txt)
        parts = [p for p in u.path.split("/") if p]
        i = parts.index("datasets")
        owner, dataset = parts[i+1], parts[i+2]
        slug = f"{owner}/{dataset}"
        csv_name = parse_qs(u.query).get("select", [None])[0] or fallback_file
        return slug, csv_name
    if re.match(r"^[a-z0-9_.-]+/[a-z0-9_.-]+$", txt, flags=re.I):
        return txt, fallback_file
    raise ValueError("Invalid Kaggle dataset format. Use owner/dataset or paste the dataset URL.")

def load_kaggle_dataset(slug: str, csv_name: str | None = None) -> pd.DataFrame:
    """
    Downloads a CSV from a Kaggle dataset using the Kaggle API and returns a DataFrame.
    Requires env vars KAGGLE_USERNAME and KAGGLE_KEY (or ~/.kaggle/kaggle.json).
    """
    api.authenticate()  # reads env or ~/.kaggle/kaggle.json
    files = api.dataset_list_files(slug).files
    names = [f.name for f in files]
    if not names:
        raise ValueError("No files in Kaggle dataset.")
    if csv_name is None:
        csvs = [n for n in names if n.lower().endswith(".csv")]
        if not csvs:
            raise ValueError("No CSV files in Kaggle dataset.")
        csv_name = csvs[0]
    elif csv_name not in names:
        raise ValueError(f"'{csv_name}' not found. Available: {names}")

    with tempfile.TemporaryDirectory() as tmp:
        api.dataset_download_file(slug, csv_name, path=tmp, force=True, quiet=True)
        # find the zip the Kaggle lib created
        zips = [os.path.join(tmp, p) for p in os.listdir(tmp) if p.endswith(".zip")]
        if not zips:
            raise RuntimeError("Kaggle download did not produce a .zip")
        with zipfile.ZipFile(zips[0]) as zf:
            zf.extractall(tmp)
        csv_path = os.path.join(tmp, csv_name)
        if not os.path.exists(csv_path):
            # fallback: first CSV found
            csvs = [os.path.join(tmp, p) for p in os.listdir(tmp) if p.lower().endswith(".csv")]
            if not csvs:
                raise RuntimeError("Archive did not contain a CSV.")
            csv_path = csvs[0]
        return pd.read_csv(csv_path, low_memory=False)
