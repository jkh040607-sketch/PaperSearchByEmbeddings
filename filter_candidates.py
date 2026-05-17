import argparse
import json
import re
from pathlib import Path


DEFAULT_INPUT = Path("outputs/openreview_ml/processed/ml_conferences_openreview.json")
DEFAULT_OUTPUT = Path("outputs/openreview_ml/processed/candidates.json")
DEFAULT_PHRASES = Path("examples/phrases.txt")


def load_phrases(path):
    phrases = []
    for line in path.read_text(encoding="utf-8").splitlines():
        phrase = line.strip()
        if not phrase or phrase.startswith("#"):
            continue
        phrases.append(phrase)
    if not phrases:
        raise ValueError(f"No phrases found in {path}")
    return phrases


def paper_text(paper):
    keywords = paper.get("keywords") or []
    if isinstance(keywords, str):
        keywords = [keywords]
    values = [
        paper.get("title", ""),
        paper.get("abstract", ""),
        paper.get("primary_area", ""),
        " ".join(str(keyword) for keyword in keywords),
    ]
    return "\n".join(str(value) for value in values if value).lower()


def phrase_pattern(phrase):
    escaped = re.escape(phrase.lower())
    return re.compile(rf"(?<![A-Za-z0-9]){escaped}(?![A-Za-z0-9])")


def score_paper(paper, patterns):
    text = paper_text(paper)
    matches = [phrase for phrase, pattern in patterns if pattern.search(text)]
    return len(matches), matches


def main():
    parser = argparse.ArgumentParser(description="Filter a paper corpus with a configurable phrase list.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input paper JSON file.")
    parser.add_argument("--phrases-file", type=Path, default=DEFAULT_PHRASES, help="One search phrase per line.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output candidate JSON file.")
    parser.add_argument("--min-score", type=int, default=1, help="Minimum number of phrase matches.")
    parser.add_argument("--limit", type=int, default=0, help="Optional maximum number of candidates to write.")
    args = parser.parse_args()

    papers = json.loads(args.input.read_text(encoding="utf-8"))
    phrases = load_phrases(args.phrases_file)
    patterns = [(phrase, phrase_pattern(phrase)) for phrase in phrases]

    candidates = []
    for paper in papers:
        score, matches = score_paper(paper, patterns)
        if score < args.min_score:
            continue
        item = dict(paper)
        item["keyword_score"] = score
        item["keyword_matches"] = matches
        candidates.append(item)

    candidates.sort(key=lambda item: (item["keyword_score"], item.get("year") or 0), reverse=True)
    if args.limit > 0:
        candidates = candidates[: args.limit]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(candidates, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Loaded papers: {len(papers)}")
    print(f"Matched candidates: {len(candidates)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
