# PaperSearchByEmbeddings

Semantic paper search over OpenReview metadata with local or OpenAI embeddings.

The default setup uses the local `Qwen/Qwen3-Embedding-0.6B` model, so paper titles and abstracts can be embedded without sending text to an API. The project can crawl OpenReview metadata, build a candidate corpus, and run topic-based semantic search with JSON, CSV, Markdown, and BibTeX outputs.

## Features

- Crawl papers from OpenReview by venue id.
- Crawl recent ICLR, ICML, and NeurIPS papers from OpenReview.
- Search by free-text query or example papers.
- Cache embeddings automatically.
- Use local SentenceTransformer models, defaulting to `Qwen/Qwen3-Embedding-0.6B`.
- Optionally use OpenAI embeddings through the OpenAI SDK.
- Keep generated corpora, caches, and search outputs out of git.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

If Hugging Face downloads are slow, try a mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1
```

Download the default local embedding model:

```bash
.venv/bin/hf download Qwen/Qwen3-Embedding-0.6B
```

After the model is cached, you can run offline:

```bash
export HF_HUB_OFFLINE=1
```

## Single-Venue Search

Fetch one OpenReview venue:

```bash
.venv/bin/python crawl.py
```

Search the generated paper file:

```bash
.venv/bin/python search_cli.py "large language model agents tool use reasoning" \
  --papers iclr2026_papers.json \
  --top-k 50 \
  --show 10 \
  --output results.json
```

Use another local SentenceTransformer model:

```bash
.venv/bin/python search_cli.py "diffusion policies for continuous control" \
  --papers iclr2026_papers.json \
  --local-model BAAI/bge-m3
```

## Multi-Conference Pipeline

The multi-conference pipeline writes generated files under `outputs/` by default.

1. Crawl OpenReview data:

```bash
.venv/bin/python crawl_ml_conferences.py
```

2. Build a candidate set from keyword phrases:

```bash
.venv/bin/python filter_candidates.py \
  --phrases-file examples/phrases.txt
```

3. Run topic-based embedding search:

```bash
HF_HUB_OFFLINE=1 .venv/bin/python topic_search.py \
  --topics examples/topics.json \
  --top-k 80 \
  --show-per-topic 25 \
  --shortlist-size 100 \
  --local-batch-size 32 \
  --local-max-seq-length 512
```

Expected outputs:

```text
outputs/topic_search/topic_results.json
outputs/topic_search/topic_results.csv
outputs/topic_search/shortlist.json
outputs/topic_search/search_report.md
outputs/topic_search/bib/shortlist.bib
```

To write generated files somewhere else, pass `--output-root` to the crawl and search scripts. Pass `--input` and `--output` to `filter_candidates.py` when using a custom corpus path.

## OpenAI Embeddings

Local embeddings are the default. To use OpenAI embeddings, configure the OpenAI SDK in your environment and run with `--model openai`.

```bash
.venv/bin/python search_cli.py "retrieval augmented generation evaluation benchmarks" \
  --papers iclr2026_papers.json \
  --model openai
```

## Repository Hygiene

The repository intentionally ignores virtual environments, local `.env` files, generated OpenReview corpora, embedding caches, search outputs, and downloaded PDFs. GitHub commits should contain source code, documentation, and lightweight examples only.
