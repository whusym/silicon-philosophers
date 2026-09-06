#!/usr/bin/env python3
"""
Chunked OpenRouter batch runner for Grok 4.3.

OpenRouter rejects batches > 5000 requests, so this splits the full
philosopher×question grid into chunks of 5000, submits each, polls, and
merges results into one combined JSON.

Usage:
  export OPENROUTER_API_KEY=...
  # Prefer a local --data-dir with full demographics (private; not in git).
  # load_data() strips any human `responses` fields before prompting.
  python 3c_run_grok43_chunked_batch.py submit --data-dir /tmp/grok43_demo_data --output-dir /tmp/grok43_demo_full
  python 3c_run_grok43_chunked_batch.py status --output-dir /tmp/grok43_demo_full
  python 3c_run_grok43_chunked_batch.py collect --output-dir /tmp/grok43_demo_full
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import importlib.util

CHUNK_SIZE = 5000
POLL_SECONDS = 30


def load_eval_module(eval_script: Optional[Path] = None):
    # Default Grok 4.3; pass --eval-script for other models (e.g. grok-4.6).
    path = eval_script or (Path(__file__).resolve().parent / "3b_eval_openrouter_grok43.py")
    spec = importlib.util.spec_from_file_location("grok_eval", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def build_all_requests(mod, data_dir: Path, limit: Optional[int]):
    philosophers, questions = mod.load_data(data_dir)
    mapping: Dict[str, Any] = {}
    requests_payload: List[Dict[str, Any]] = []
    count = 0
    for philosopher in philosophers:
        phil_name = philosopher.get("name", "Unknown")
        for qkey, options in questions.items():
            if limit is not None and count >= limit:
                return mapping, requests_payload
            custom_id = f"req_{count}"
            prompt = mod.build_full_prompt(philosopher, qkey, options)
            mapping[custom_id] = {
                "philosopher_name": phil_name,
                "question": qkey,
                "options": options,
                "philosopher": mod.philosopher_snapshot(philosopher),
            }
            body = {
                "messages": [{"role": "user", "content": prompt}],
                "temperature": mod.TEMPERATURE,
                "max_tokens": mod.MAX_TOKENS,
                "reasoning": {"enabled": mod.REASONING_ENABLED},
            }
            requests_payload.append({"custom_id": custom_id, "body": body})
            count += 1
    return mapping, requests_payload


def chunked_dir(output_dir: Path) -> Path:
    d = output_dir / "openrouter_batch_chunks"
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_submit(
    data_dir: Path,
    output_dir: Path,
    limit: Optional[int],
    eval_script: Optional[Path] = None,
) -> None:
    import requests

    mod = load_eval_module(eval_script)
    api_key = mod.require_api_key()
    mapping, payload = build_all_requests(mod, data_dir, limit)
    out = chunked_dir(output_dir)

    with open(out / "mapping.json", "w") as f:
        json.dump(mapping, f)
    with open(out / "all_requests_meta.json", "w") as f:
        json.dump({"n_requests": len(payload), "chunk_size": CHUNK_SIZE}, f, indent=2)

    chunks = [
        payload[i : i + CHUNK_SIZE] for i in range(0, len(payload), CHUNK_SIZE)
    ]
    print(f"Total requests: {len(payload)}")
    print(f"Chunks: {len(chunks)} (max {CHUNK_SIZE} each)")

    # Resume support: keep already-submitted chunk ids if present.
    ids_path = out / "batch_ids.json"
    existing = {}
    if ids_path.exists():
        try:
            existing = {
                b["chunk"]: b for b in json.load(open(ids_path)).get("batches", [])
            }
        except Exception:
            existing = {}

    batch_ids = []
    for ci, chunk in enumerate(chunks):
        if ci in existing:
            print(f"\nSkipping chunk {ci+1}/{len(chunks)} (already submitted: {existing[ci]['id']})")
            batch_ids.append(existing[ci])
            continue

        payload_json = (
            '{"endpoint":"/v1/chat/completions","model":'
            + json.dumps(mod.MODEL_ID)
            + ',"requests":'
            + json.dumps(chunk, separators=(",", ":"))
            + "}"
        )
        print(f"\nSubmitting chunk {ci+1}/{len(chunks)} ({len(chunk)} requests)...")
        # Retry on 429 in-flight limits.
        for attempt in range(60):
            resp = requests.post(
                mod.OPENROUTER_BATCH_URL,
                headers=mod.openrouter_headers(api_key),
                data=payload_json,
                timeout=600,
            )
            if resp.status_code == 429:
                print(f"  429 in-flight limit; waiting 60s (attempt {attempt+1}/60)...")
                time.sleep(60)
                continue
            break
        if resp.status_code >= 400:
            # Persist progress before dying.
            with open(ids_path, "w") as f:
                json.dump({"batches": batch_ids}, f, indent=2)
            raise SystemExit(
                f"Chunk {ci} submit failed ({resp.status_code}): {resp.text[:500]}"
            )
        created = resp.json()
        batch_id = created.get("id")
        if not batch_id:
            with open(ids_path, "w") as f:
                json.dump({"batches": batch_ids}, f, indent=2)
            raise SystemExit(f"Chunk {ci}: no batch id in {created}")
        batch_ids.append(
            {
                "chunk": ci,
                "id": batch_id,
                "n_requests": len(chunk),
                "raw": created,
            }
        )
        print(f"  chunk {ci}: batch_id={batch_id}")
        # Save after every successful chunk so a later 429 does not lose IDs.
        with open(ids_path, "w") as f:
            json.dump({"batches": batch_ids}, f, indent=2)
        time.sleep(1)

    print(f"\nSaved {ids_path}")
    print("Next: python 3c_run_grok43_chunked_batch.py status --output-dir ...")

def cmd_status(output_dir: Path, eval_script: Optional[Path] = None) -> None:
    import requests

    mod = load_eval_module(eval_script)
    api_key = mod.require_api_key()
    out = chunked_dir(output_dir)
    meta_path = out / "batch_ids.json"
    if not meta_path.exists():
        raise SystemExit(f"Missing {meta_path}; run submit first")
    meta = json.load(open(meta_path))
    all_done = True
    for b in meta["batches"]:
        resp = requests.get(
            f"{mod.OPENROUTER_BATCH_URL}/{b['id']}",
            headers=mod.openrouter_headers(api_key),
            timeout=60,
        )
        if resp.status_code >= 400:
            print(f"chunk {b['chunk']}: status check failed {resp.status_code} {resp.text[:200]}")
            all_done = False
            continue
        batch = resp.json()
        status = batch.get("status")
        print(f"chunk {b['chunk']}: id={b['id']} status={status}")
        for k in ("request_counts", "completed", "failed", "total"):
            if k in batch:
                print(f"  {k}: {batch[k]}")
        if status not in {"completed", "failed", "cancelled", "expired", "ended"}:
            all_done = False
    print("ALL COMPLETE" if all_done else "STILL RUNNING")


def _extract_content(item: Dict[str, Any]):
    if item.get("error"):
        return None, item["error"]
    response = item.get("response") or item.get("body") or item
    if not isinstance(response, dict):
        return None, "unexpected response shape"
    choices = response.get("choices")
    if choices:
        return choices[0].get("message", {}).get("content"), None
    body = response.get("body")
    if isinstance(body, dict):
        choices = body.get("choices") or []
        if choices:
            return choices[0].get("message", {}).get("content"), None
    if response.get("output_text"):
        return response["output_text"], None
    return None, "no content found"


def score_option(parsed: List[str], position: str) -> Optional[float]:
    """Map stance-toward-position options to 0/0.25/0.5/0.75/1."""
    if not parsed:
        return None
    text = " ".join(parsed).lower()
    pos = position.lower()
    if "agnostic" in text or "undecided" in text:
        return None
    # check ordered from strong accept to reject
    if f"accept: {pos}" in text or text.strip() == f"accept: {pos}":
        return 1.0
    if f"lean towards: {pos}" in text or f"lean toward: {pos}" in text:
        return 0.75
    if f"neutral towards: {pos}" in text or f"neutral toward: {pos}" in text:
        return 0.5
    if f"lean against: {pos}" in text:
        return 0.25
    if f"reject: {pos}" in text:
        return 0.0
    # fallback: first item exact-ish
    first = parsed[0].lower()
    mapping = {
        f"accept: {pos}": 1.0,
        f"lean towards: {pos}": 0.75,
        f"lean toward: {pos}": 0.75,
        f"neutral towards: {pos}": 0.5,
        f"neutral toward: {pos}": 0.5,
        f"lean against: {pos}": 0.25,
        f"reject: {pos}": 0.0,
    }
    return mapping.get(first)


def cmd_collect(
    output_dir: Path, wait: bool = True, eval_script: Optional[Path] = None
) -> None:
    import requests

    mod = load_eval_module(eval_script)
    api_key = mod.require_api_key()
    out = chunked_dir(output_dir)
    meta = json.load(open(out / "batch_ids.json"))
    mapping = json.load(open(out / "mapping.json"))

    # Poll until all terminal if requested
    while True:
        statuses = []
        for b in meta["batches"]:
            resp = requests.get(
                f"{mod.OPENROUTER_BATCH_URL}/{b['id']}",
                headers=mod.openrouter_headers(api_key),
                timeout=120,
            )
            resp.raise_for_status()
            batch = resp.json()
            statuses.append(batch.get("status"))
            print(f"chunk {b['chunk']}: {batch.get('status')}")
        terminal = {"completed", "failed", "cancelled", "expired", "ended"}
        if all(s in terminal for s in statuses):
            break
        if not wait:
            raise SystemExit("Not all batches complete yet")
        time.sleep(POLL_SECONDS)

    combined: List[Dict[str, Any]] = []
    success = failed = 0
    scored_rows = []

    for b in meta["batches"]:
        resp = requests.get(
            f"{mod.OPENROUTER_BATCH_URL}/{b['id']}",
            headers=mod.openrouter_headers(api_key),
            timeout=300,
        )
        resp.raise_for_status()
        batch = resp.json()
        with open(out / f"raw_chunk_{b['chunk']}.json", "w") as f:
            json.dump(batch, f)
        raw_results = (
            batch.get("results") or batch.get("output") or batch.get("data") or []
        )
        if isinstance(raw_results, dict):
            raw_results = raw_results.get("results") or raw_results.get("data") or []
        if not raw_results:
            url = batch.get("output_file_url") or batch.get("results_url")
            if url:
                r = requests.get(url, headers=mod.openrouter_headers(api_key), timeout=300)
                r.raise_for_status()
                text = r.text.strip()
                raw_results = (
                    json.loads(text)
                    if text.startswith("[")
                    else [json.loads(line) for line in text.splitlines() if line.strip()]
                )
        print(f"chunk {b['chunk']}: {len(raw_results)} results")

        for item in raw_results:
            custom_id = item.get("custom_id") or item.get("customId")
            meta_row = mapping.get(custom_id)
            if not meta_row:
                failed += 1
                continue
            content, err = _extract_content(item)
            parsed = mod.parse_response_list(content or "") if content else []
            ok, msg = (
                mod.validate_response(parsed, meta_row["options"])
                if content
                else (False, "no content")
            )
            if ok:
                success += 1
            else:
                failed += 1
                if not err:
                    err = msg

            # question key format: "{base}: {position}"
            q = meta_row["question"]
            position = q.split(": ", 1)[1] if ": " in q else q
            score = score_option(parsed, position) if ok else None

            entry = {
                "model": mod.MODEL_ID,
                "philosopher": meta_row["philosopher"],
                "question": meta_row["question"],
                "response": {
                    "parsed": parsed,
                    "raw": content or "",
                    "success": bool(ok),
                    "error": None if ok else (err if isinstance(err, str) else json.dumps(err)),
                    "score": score,
                    "custom_id": custom_id,
                },
            }
            combined.append(entry)
            scored_rows.append(
                {
                    "philosopher": meta_row["philosopher_name"],
                    "question": meta_row["question"],
                    "score": score,
                    "success": bool(ok),
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    combined_path = output_dir / f"{mod.MODEL_LABEL}_combined.json"
    scored_path = output_dir / f"{mod.MODEL_LABEL}_scored.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f)
    with open(scored_path, "w") as f:
        json.dump(scored_rows, f)

    print("\n" + "=" * 72)
    print("CHUNKED COLLECT COMPLETE")
    print(f"Results: {len(combined)}  success≈{success}  failed/invalid≈{failed}")
    print(f"Combined: {combined_path}")
    print(f"Scored:   {scored_path}")
    print("=" * 72)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("action", choices=["submit", "status", "collect"])
    p.add_argument("--data-dir", type=Path, default=Path("full_data_reconstructed"))
    p.add_argument("--output-dir", type=Path, default=Path("llm_responses_grok-4.3"))
    p.add_argument(
        "--eval-script",
        type=Path,
        default=None,
        help="Eval module path (default: 3b_eval_openrouter_grok43.py)",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-wait", action="store_true")
    args = p.parse_args()
    if args.action == "submit":
        cmd_submit(args.data_dir, args.output_dir, args.limit, args.eval_script)
    elif args.action == "status":
        cmd_status(args.output_dir, args.eval_script)
    else:
        cmd_collect(args.output_dir, wait=not args.no_wait, eval_script=args.eval_script)


if __name__ == "__main__":
    main()
