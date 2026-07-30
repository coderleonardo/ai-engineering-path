# AI Engineering Notes

Personal coursework repo for Data Science Academy's **AI Engineering** track: notes, worked examples,
and reference implementations built while going through the course. It is not a shipped application —
there is no build, lint, or test suite, just content organized so it can be searched and reused later.

## Layout

- **[`gen-ai-and-llms/`](./gen-ai-and-llms/)** — the active course track: generative AI and LLMs. Each
  numbered subfolder is one module (transformer architectures, prompt engineering, fine-tuning/QLoRA,
  LangChain, vector databases, RAG, deployment, ...). Start with
  [`gen-ai-and-llms/README.md`](./gen-ai-and-llms/README.md) — it's a cookbook that walks through every
  module with a short concept summary, links to the notes/code, and where each idea originally comes
  from. [`gen-ai-and-llms/q&a.md`](./gen-ai-and-llms/q&a.md) has quick Q&A for review, grounded in and
  linked back to those same notes.
- **`computer-vision/`** — placeholder for a future course track; empty for now.

## Environment

- Python 3.11 (`.python-version`), dependencies managed per-track with [`uv`](https://docs.astral.sh/uv/)
  (each track has its own `pyproject.toml`; lockfiles are intentionally gitignored here since this is a
  notes repo, not a distributable package).
- Install/sync deps from inside a track's folder: `uv sync`. Run a script or notebook kernel in that env:
  `uv run <cmd>` or `uv run jupyter lab`.
- API keys used by exercises live in a gitignored `.env` (see each track's own docs for which variables
  it expects).
