import argparse
import json
import time
from pathlib import Path

import requests


DEFAULT_OUTPUT_ROOT = Path("outputs/openreview_ml")

VENUES = [
    {"conference": "ICLR", "year": 2026, "venue_id": "ICLR.cc/2026/Conference", "api": "api2"},
    {"conference": "ICLR", "year": 2025, "venue_id": "ICLR.cc/2025/Conference", "api": "api2"},
    {"conference": "ICLR", "year": 2024, "venue_id": "ICLR.cc/2024/Conference", "api": "api2"},
    {"conference": "ICLR", "year": 2023, "venue_id": "ICLR.cc/2023/Conference", "api": "api1"},
    {"conference": "ICML", "year": 2025, "venue_id": "ICML.cc/2025/Conference", "api": "api2"},
    {"conference": "ICML", "year": 2024, "venue_id": "ICML.cc/2024/Conference", "api": "api2"},
    {"conference": "ICML", "year": 2023, "venue_id": "ICML.cc/2023/Conference", "api": "api2"},
    {"conference": "NeurIPS", "year": 2025, "venue_id": "NeurIPS.cc/2025/Conference", "api": "api2"},
    {"conference": "NeurIPS", "year": 2024, "venue_id": "NeurIPS.cc/2024/Conference", "api": "api2"},
    {"conference": "NeurIPS", "year": 2023, "venue_id": "NeurIPS.cc/2023/Conference", "api": "api2"},
]


def unwrap(value, default=None):
    if isinstance(value, dict) and "value" in value:
        return value.get("value", default)
    if value is None:
        return default
    return value


def api_base(api_name):
    if api_name == "api1":
        return "https://api.openreview.net/notes"
    if api_name == "api2":
        return "https://api2.openreview.net/notes"
    raise ValueError(f"Unknown API: {api_name}")


def fetch_venue(venue_id, api_name, limit=1000, sleep=0.5):
    url = api_base(api_name)
    offset = 0
    notes = []
    while True:
        params = {
            "content.venueid": venue_id,
            "details": "replyCount,invitation",
            "limit": limit,
            "offset": offset,
            "sort": "number:desc",
        }
        response = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=60)
        response.raise_for_status()
        batch = response.json().get("notes", [])
        if not batch:
            break
        notes.extend(batch)
        print(f"  fetched {len(batch)} notes, total={len(notes)}")
        if len(batch) < limit:
            break
        offset += limit
        time.sleep(sleep)
    return notes


def normalize_note(note, venue):
    content = note.get("content", {})
    paper_id = note.get("id") or note.get("forum")
    title = unwrap(content.get("title"), "")
    abstract = unwrap(content.get("abstract"), "")
    authors = unwrap(content.get("authors"), [])
    keywords = unwrap(content.get("keywords"), [])
    primary_area = unwrap(content.get("primary_area"), "") or unwrap(content.get("subject_areas"), "")
    venue_text = unwrap(content.get("venue"), "")
    venue_id = unwrap(content.get("venueid"), venue["venue_id"])
    pdf_url = f"https://openreview.net/pdf?id={paper_id}" if paper_id else ""
    forum_url = f"https://openreview.net/forum?id={paper_id}" if paper_id else ""

    if isinstance(authors, str):
        authors = [authors]
    if isinstance(keywords, str):
        keywords = [keywords]
    if isinstance(primary_area, list):
        primary_area = ", ".join(str(item) for item in primary_area)

    return {
        "id": paper_id,
        "number": note.get("number"),
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "keywords": keywords,
        "primary_area": primary_area,
        "venue": venue_text,
        "venue_id": venue_id,
        "conference": venue["conference"],
        "year": venue["year"],
        "source_api": venue["api"],
        "forum_url": forum_url,
        "pdf_url": pdf_url,
    }


def dedupe_papers(papers):
    seen = set()
    deduped = []
    for paper in papers:
        key = paper.get("id") or paper.get("title", "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(paper)
    return deduped


def main():
    parser = argparse.ArgumentParser(description="Crawl recent ICLR/ICML/NeurIPS papers from OpenReview.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--sleep", type=float, default=0.5)
    args = parser.parse_args()

    raw_dir = args.output_root / "raw"
    processed_dir = args.output_root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    all_papers = []
    manifest = []
    for venue in VENUES:
        label = f"{venue['conference']}_{venue['year']}"
        print(f"\nFetching {label}: {venue['venue_id']} ({venue['api']})")
        notes = fetch_venue(venue["venue_id"], venue["api"], limit=args.limit, sleep=args.sleep)
        raw_file = raw_dir / f"{label}.json"
        raw_file.write_text(json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8")

        papers = [normalize_note(note, venue) for note in notes]
        all_papers.extend(papers)
        manifest.append({**venue, "count": len(papers), "raw_file": str(raw_file)})
        print(f"Saved {len(papers)} normalized records for {label}")

    all_papers = dedupe_papers(all_papers)
    combined_file = processed_dir / "ml_conferences_openreview.json"
    manifest_file = processed_dir / "ml_conferences_manifest.json"
    combined_file.write_text(json.dumps(all_papers, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_file.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nCombined papers: {len(all_papers)}")
    print(f"Saved to: {combined_file}")
    print(f"Manifest: {manifest_file}")


if __name__ == "__main__":
    main()
