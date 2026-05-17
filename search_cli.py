import argparse

from search import PaperSearcher


def main():
    parser = argparse.ArgumentParser(description="Search papers by semantic similarity.")
    parser.add_argument("query", help="Text query describing the papers you want to find.")
    parser.add_argument("--papers", default="iclr2026_papers.json", help="Path to the paper JSON file.")
    parser.add_argument("--model", choices=["local", "openai"], default="local", help="Embedding model type.")
    parser.add_argument(
        "--local-model",
        default=PaperSearcher.DEFAULT_LOCAL_MODEL,
        help="SentenceTransformer model to use when --model local.",
    )
    parser.add_argument("--local-batch-size", type=int, default=64, help="Local embedding batch size.")
    parser.add_argument("--local-max-seq-length", type=int, default=1024, help="Local model token truncation length.")
    parser.add_argument("--top-k", type=int, default=50, help="Number of results to rank.")
    parser.add_argument("--show", type=int, default=10, help="Number of results to print.")
    parser.add_argument("--output", default="results.json", help="Where to save the ranked results.")
    parser.add_argument("--force", action="store_true", help="Recompute embeddings instead of using cache.")
    args = parser.parse_args()

    searcher = PaperSearcher(
        args.papers,
        model_type=args.model,
        local_model_name=args.local_model,
        local_batch_size=args.local_batch_size,
        local_max_seq_length=args.local_max_seq_length,
    )
    searcher.compute_embeddings(force=args.force)
    results = searcher.search(query=args.query, top_k=args.top_k)
    searcher.display(results, n=args.show)
    searcher.save(results, args.output)


if __name__ == "__main__":
    main()
