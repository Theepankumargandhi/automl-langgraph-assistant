MCP Add-On: Orchestrating AutoML with Model Context Protocol
TL;DR

This add-on exposes your AutoML pipeline (LangGraph + Chroma RAG + Streamlit app) as MCP tools so any MCP-compatible client (e.g., Claude Desktop or internal agent runners) can: load data → profile → RAG-retrieve rules → generate/repair code → execute safely → evaluate → export Markdown/PDF → fetch plots — without using the Streamlit UI.

What you’ll add

An MCP server that wraps your existing pipeline and artifacts.

A set of tools:

load_dataset – load from upload/path/Kaggle.

profile_dataset – infer schema, task, missingness.

ingest_rules – (re)index rules/*.md into Chroma.

run_automl – end-to-end AutoML (LLM vs baseline + guardrails).

export_report – build run_summary.pdf from Markdown + plots.

list_artifacts / get_artifact – enumerate & fetch images/MD/PDF as resources.

A resource interface so plots (step_*_plot.png) and reports are retrievable via URIs instead of raw blobs.

Minimal security/config: reuse your env flags (ALLOW_IO, ALLOW_TUNING, ALLOWED_DATA_DIR) + existing keys.

Why this matters

Decoupled UI: Streamlit is now just one client; the pipeline becomes a reusable service.

Real-world ops: CI can smoke-test rules & runs; bots can schedule daily profiles and export PDFs.

Recruiter impact: Shows platform thinking (protocol tools, resources, containers, CI), not just a demo app.

Where it fits in your current flow

Today (Streamlit):

Load CSV/Kaggle → 2) Profile → 3) RAG rules (Chroma) → 4) Plan/codegen/auto-repair →

Safe execute → 6) Evaluate & choose (LLM vs baseline) → 7) Produce summary.md + plots → 8) Export PDF.

With MCP:
Each numbered step is available as a tool call; artifacts are resources you can fetch from any MCP client.

Tools (contract overview)
Tool	Inputs	Output (summary)	Typical use
load_dataset	source (upload/path/kaggle), path_or_slug, csv_name?	rows, cols, preview hash	Provide data to the session
profile_dataset	—	schema, dtypes, numeric/cat counts, missingness, inferred task	Inspect before training
ingest_rules	—	number of files indexed, chunks added	Refresh Chroma after editing rules/*.md
run_automl	target_column	metrics (accuracy/F1/RMSE/R²), winner (ai/baseline), artifact names	Core AutoML pipeline
export_report	—	resource URI for run_summary.pdf	Shareable PDF report
list_artifacts	glob?	list of artifact names & URIs	Discover files created in the run
get_artifact	name	resource stream/URI	Download plot/MD/PDF

Notes
• Tools call your existing LangGraph create_graph().invoke(...), RAG indexer, and ReportLab exporter.
• Resources serve files from a dedicated artifacts/ (or working) directory, not arbitrary paths.

What you’ll add to the repo (lightweight structure)
mcp/
  server.py          # registers tools, exposes resources, loads .env
  tools.py           # thin wrappers calling your existing app modules
  resources.py       # safe file/resource serving from artifacts dir
docs/
  mcp_quickstart.md  # this README section (or link)


Keep your current app code unchanged; the MCP layer is a thin adapter.

Prerequisites

Your existing project environment working end-to-end (Streamlit run, PDF export ok).

MCP-compatible client (e.g., Claude Desktop) or an internal agent runner that can call MCP tools.

Environment variables you already use (e.g., OPENAI_API_KEY or other LLM provider, KAGGLE_USERNAME, KAGGLE_KEY, ALLOW_IO, ALLOW_TUNING, ALLOWED_DATA_DIR).

Setup (high level)

Create mcp/ with server.py, tools.py, resources.py (thin wrappers around your current Python functions).

Point the server at your artifacts directory (e.g., the app working dir) for plots and PDFs.

Expose each step as a tool; register artifacts as resources.

Run the MCP server locally; add it in your MCP client settings (e.g., “Custom server” with command + env).

(Optional) Add a docker-compose.mcp.yml that runs:

your MCP server

Chroma with a persisted volume

(optionally) a one-shot CI smoke test service

Prompt pack (for MCP chat clients)

These are copy-paste prompts a beginner can use. The client will call the tools under the hood.

Profile first

Use the `load_dataset` tool with source="path" and path_or_slug="/absolute/path/to/data.csv".
Then call `profile_dataset` and summarize the inferred task, numeric/categorical counts, and missingness.


Index rules (RAG)

Call `ingest_rules` to re-index the markdown files under rules/.
Tell me how many files/chunks were added.


Train end-to-end

Run `run_automl` with target_column="<YOUR_TARGET>".
Report accuracy/F1 (for classification) or RMSE/R² (for regression), the winning model origin (ai or baseline), and list the produced artifacts.


Export and fetch

Call `export_report` to build the PDF, then list artifacts with `list_artifacts`.
Finally, fetch the PDF and one plot via `get_artifact`.

Security & config (kept simple)

Guardrails: ALLOW_IO, ALLOW_TUNING, ALLOWED_DATA_DIR still apply; MCP tools should respect them.

Resources: Only serve files from a whitelisted folder (no absolute path traversal).

Secrets: Keep keys in .env; clients don’t send credentials.

CI smoke test (optional but recommended)

In CI, call ingest_rules → load_dataset (small sample) → run_automl (fast params).

Upload run_summary.pdf as a CI artifact to prove end-to-end health.

Time & resume impact

MCP server + tools + resources: ~0.5–1 day

Docs + demo GIF: ~0.5 day

Docker/Compose + optional CI smoke test: ~0.5–1 day
Impact: High — you demonstrate protocol tooling, service boundaries, artifact governance, and cross-client orchestration.

Troubleshooting (quick hits)

No plots in PDF? Ensure images exist in the artifacts directory before calling export_report.

Client can’t fetch resources? Verify the resource URI roots and that files live under the whitelisted folder.

Rules not applied? Re-run ingest_rules after editing rules/*.md.

Large CSVs slow? Keep MAX_CSV_BYTES sensible; MCP tools should return summaries, not whole data.

Non-goals (for now)

Long-running background jobs (can be added later with job_id + get_status).

Auth/quotas (simple .env-based gating is fine initially).