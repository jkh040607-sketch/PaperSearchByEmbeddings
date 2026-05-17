import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from search import PaperSearcher


DEFAULT_PAPERS = Path("outputs/openreview_ml/processed/candidates.json")
DEFAULT_TOPICS = Path("examples/topics.json")
DEFAULT_OUTPUT_ROOT = Path("outputs/topic_search")


def safe_text(value):
    if value is None:
        return ""
    if isinstance(value, list):
        return ", ".join(str(item) for item in value)
    return str(value)


def load_topics(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    topics = data.get("topics", data) if isinstance(data, dict) else data
    if not isinstance(topics, list):
        raise ValueError("Topics file must contain a list or an object with a 'topics' list.")
    for index, topic in enumerate(topics, 1):
        for field in ("id", "title", "query"):
            if not topic.get(field):
                raise ValueError(f"Topic {index} is missing '{field}'.")
    return topics


def bib_key(paper):
    authors = paper.get("authors") or []
    first = "paper"
    if authors:
        first = re.sub(r"[^A-Za-z0-9]", "", str(authors[0]).split()[-1]).lower() or "paper"
    first_title_word = (paper.get("title") or "work").split()[0]
    title_word = re.sub(r"[^A-Za-z0-9]", "", first_title_word.lower()) or "work"
    year = paper.get("year", "na")
    return f"{first}{year}{title_word}"


def write_bib(papers, path):
    used = Counter()
    entries = []
    for paper in papers:
        base_key = bib_key(paper)
        used[base_key] += 1
        key = base_key if used[base_key] == 1 else f"{base_key}{used[base_key]}"
        authors = " and ".join(paper.get("authors") or [])
        booktitle = f"{paper.get('conference', '')} {paper.get('year', '')}".strip()
        fields = {
            "title": paper.get("title", ""),
            "author": authors,
            "booktitle": booktitle,
            "year": str(paper.get("year", "")),
            "url": paper.get("forum_url", ""),
        }
        body = "\n".join(f"  {name} = {{{value}}}," for name, value in fields.items() if value)
        entries.append(f"@inproceedings{{{key},\n{body}\n}}")
    path.write_text("\n\n".join(entries) + ("\n" if entries else ""), encoding="utf-8")


def result_row(topic, rank, result):
    paper = result["paper"]
    return {
        "topic_id": topic["id"],
        "topic_title": topic["title"],
        "rank": rank,
        "similarity": round(result["similarity"], 6),
        "title": paper.get("title", ""),
        "conference": paper.get("conference", ""),
        "year": paper.get("year", ""),
        "primary_area": safe_text(paper.get("primary_area", "")),
        "forum_url": paper.get("forum_url", ""),
        "pdf_url": paper.get("pdf_url", ""),
        "abstract": paper.get("abstract", ""),
    }


def summarize_corpus(papers):
    counts = Counter((paper.get("conference", ""), paper.get("year", "")) for paper in papers)
    return [
        {"conference": conference, "year": year, "count": count}
        for (conference, year), count in sorted(counts.items(), key=lambda item: (item[0][0], item[0][1]))
    ]


def write_markdown_report(topic_rows, shortlist, summary, output_file, model_name):
    lines = [
        "# Topic Search Report",
        "",
        f"Embedding model: `{model_name}`",
        "",
        "## Corpus Coverage",
        "",
    ]
    for item in summary:
        label = f"{item['conference']} {item['year']}".strip() or "Unknown venue"
        lines.append(f"- {label}: {item['count']} papers")
    lines.extend(["", "## Cross-Topic Shortlist", ""])

    for index, paper in enumerate(shortlist, 1):
        topics = ", ".join(sorted(paper["topics"]))
        abstract = (paper.get("abstract") or "").strip()
        lines.extend(
            [
                f"### {index}. {paper.get('title', 'Untitled')}",
                "",
                f"- Venue: {paper.get('conference', '')} {paper.get('year', '')}".strip(),
                f"- Best similarity: {paper['best_similarity']:.4f}",
                f"- Matched topics: {topics}",
                f"- OpenReview: {paper.get('forum_url', '')}",
                f"- PDF: {paper.get('pdf_url', '')}",
                "",
                abstract[:1200] + ("..." if len(abstract) > 1200 else ""),
                "",
            ]
        )

    lines.append("## Topic Results")
    lines.append("")
    for rows in topic_rows.values():
        if not rows:
            continue
        topic_title = rows[0]["topic_title"]
        lines.append(f"### {topic_title}")
        lines.append("")
        lines.append("| Rank | Score | Paper | Venue | Link |")
        lines.append("|---:|---:|---|---|---|")
        for row in rows:
            title = row["title"].replace("|", "\\|")
            venue = f"{row['conference']} {row['year']}".strip()
            lines.append(
                f"| {row['rank']} | {row['similarity']:.4f} | {title} | {venue} | [OpenReview]({row['forum_url']}) |"
            )
        lines.append("")

    output_file.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Run topic-based semantic search over a paper corpus.")
    parser.add_argument("--papers", type=Path, default=DEFAULT_PAPERS, help="Input paper JSON file.")
    parser.add_argument("--topics", type=Path, default=DEFAULT_TOPICS, help="Topic JSON file.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", choices=["local", "openai"], default="local", help="Embedding model type.")
    parser.add_argument("--local-model", default=PaperSearcher.DEFAULT_LOCAL_MODEL)
    parser.add_argument("--local-batch-size", type=int, default=64)
    parser.add_argument("--local-max-seq-length", type=int, default=1024)
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--show-per-topic", type=int, default=20)
    parser.add_argument("--shortlist-size", type=int, default=80)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    topics = load_topics(args.topics)
    searcher = PaperSearcher(
        str(args.papers),
        model_type=args.model,
        local_model_name=args.local_model,
        local_batch_size=args.local_batch_size,
        local_max_seq_length=args.local_max_seq_length,
    )
    searcher.compute_embeddings(force=args.force)

    search_dir = args.output_root
    bib_dir = args.output_root / "bib"
    search_dir.mkdir(parents=True, exist_ok=True)
    bib_dir.mkdir(parents=True, exist_ok=True)

    topic_rows = defaultdict(list)
    all_rows = []
    candidate_map = {}

    for topic in topics:
        print(f"Searching topic: {topic['id']}")
        results = searcher.search(query=topic["query"], top_k=args.top_k)
        for rank, result in enumerate(results[: args.show_per_topic], 1):
            row = result_row(topic, rank, result)
            topic_rows[topic["id"]].append(row)
            all_rows.append(row)

        for result in results:
            paper = result["paper"]
            paper_id = paper.get("id") or paper.get("title", "")
            existing = candidate_map.setdefault(
                paper_id,
                {
                    **paper,
                    "best_similarity": result["similarity"],
                    "topics": set(),
                    "topic_scores": {},
                },
            )
            existing["best_similarity"] = max(existing["best_similarity"], result["similarity"])
            existing["topics"].add(topic["id"])
            existing["topic_scores"][topic["id"]] = result["similarity"]

    shortlist = sorted(
        candidate_map.values(),
        key=lambda paper: (len(paper["topics"]), paper["best_similarity"]),
        reverse=True,
    )[: args.shortlist_size]

    serializable_shortlist = []
    for paper in shortlist:
        item = dict(paper)
        item["topics"] = sorted(item["topics"])
        serializable_shortlist.append(item)

    topic_results_file = search_dir / "topic_results.json"
    shortlist_file = search_dir / "shortlist.json"
    csv_file = search_dir / "topic_results.csv"
    report_file = search_dir / "search_report.md"
    bib_file = bib_dir / "shortlist.bib"

    topic_results_file.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    shortlist_file.write_text(json.dumps(serializable_shortlist, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_fields = [
        "topic_id",
        "topic_title",
        "rank",
        "similarity",
        "title",
        "conference",
        "year",
        "primary_area",
        "forum_url",
        "pdf_url",
        "abstract",
    ]
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(all_rows)

    write_markdown_report(
        topic_rows,
        serializable_shortlist[:30],
        summarize_corpus(searcher.papers),
        report_file,
        searcher.model_name,
    )
    write_bib(serializable_shortlist, bib_file)

    print(f"Topic results: {topic_results_file}")
    print(f"Shortlist: {shortlist_file}")
    print(f"CSV: {csv_file}")
    print(f"Markdown report: {report_file}")
    print(f"BibTeX: {bib_file}")


if __name__ == "__main__":
    main()
