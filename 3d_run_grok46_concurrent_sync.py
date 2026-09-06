#!/usr/bin/env python3
"""Concurrent sync runner for models without OpenRouter :batch (e.g. Grok 4.6)."""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_mod(path: Path):
    spec = importlib.util.spec_from_file_location("eval_mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval-script", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--min-credits-remaining", type=float, default=2.0)
    args = p.parse_args()

    mod = load_mod(args.eval_script)
    api_key = mod.require_api_key()
    philosophers, questions = mod.load_data(args.data_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_dir / "progress.json"
    combined_path = args.output_dir / f"{mod.MODEL_LABEL}_combined.json"
    scored_path = args.output_dir / f"{mod.MODEL_LABEL}_scored.json"

    completed: Dict[str, Any] = {}
    if progress_path.exists():
        completed = json.load(open(progress_path))
        print(f"Resuming with {len(completed)} completed")

    tasks = []
    for philosopher in philosophers:
        phil_name = philosopher.get("name", "Unknown")
        for qkey, options in questions.items():
            cid = mod.combo_id(phil_name, qkey)
            if cid in completed:
                continue
            tasks.append((cid, philosopher, phil_name, qkey, options))
            if args.limit is not None and len(tasks) >= args.limit:
                break
        if args.limit is not None and len(tasks) >= args.limit:
            break

    print(f"Remaining tasks: {len(tasks)}  workers={args.workers}")
    if not tasks:
        print("Nothing to do")
        return

    import requests

    def credits_remaining() -> Optional[float]:
        try:
            r = requests.get(
                "https://openrouter.ai/api/v1/key",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=30,
            )
            return float(r.json()["data"].get("limit_remaining"))
        except Exception:
            return None

    def run_one(task):
        cid, philosopher, phil_name, qkey, options = task
        prompt = mod.build_full_prompt(philosopher, qkey, options)
        t0 = time.time()
        try:
            raw = mod.chat_completion(api_key, prompt)
            parsed = mod.parse_response_list(raw)
            ok, msg = mod.validate_response(parsed, options)
            # score position after ':' in question key
            position = qkey.split(":")[-1].strip() if ":" in qkey else qkey
            score = None
            if ok and parsed:
                text = " ".join(parsed).lower()
                pos = position.lower()
                # match paper stance labels used in options
                ordered = [
                    (f"accept: {pos}", 1.0),
                    (f"lean towards: {pos}", 0.75),
                    (f"lean toward: {pos}", 0.75),
                    (f"neutral towards: {pos}", 0.5),
                    (f"neutral toward: {pos}", 0.5),
                    (f"lean against: {pos}", 0.25),
                    (f"reject: {pos}", 0.0),
                ]
                first = parsed[0].lower().strip()
                for k, v in ordered:
                    if first == k or k in text:
                        score = v
                        break
                if score is None and ("agnostic" in text or "undecided" in text):
                    score = None
            row = {
                "model": mod.MODEL_ID,
                "philosopher": mod.philosopher_snapshot(philosopher),
                "question": qkey,
                "response": {
                    "parsed": parsed if ok else [],
                    "raw": raw,
                    "success": bool(ok),
                    "error": None if ok else msg,
                    "score": score,
                    "custom_id": cid,
                    "generation_time": time.time() - t0,
                },
            }
            return cid, row, None
        except Exception as e:
            return cid, None, str(e)

    done = 0
    errors = 0
    save_every = 50
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            cid, row, err = fut.result()
            if err or row is None:
                errors += 1
                print(f"ERR {cid}: {err}")
            else:
                completed[cid] = row
                done += 1
            if (done + errors) % save_every == 0:
                with open(progress_path, "w") as f:
                    json.dump(completed, f)
                rem = credits_remaining()
                print(f"progress {done}/{len(tasks)} errors={errors} credits_remaining={rem}")
                if rem is not None and rem < args.min_credits_remaining:
                    print("Stopping early: credits low")
                    break

    with open(progress_path, "w") as f:
        json.dump(completed, f)
    combined = list(completed.values())
    scored = []
    for row in combined:
        resp = row["response"]
        scored.append(
            {
                "philosopher": row["philosopher"].get("name")
                if isinstance(row["philosopher"], dict)
                else row["philosopher"],
                "question": row["question"],
                "score": resp.get("score"),
                "success": resp.get("success"),
            }
        )
    with open(combined_path, "w") as f:
        json.dump(combined, f)
    with open(scored_path, "w") as f:
        json.dump(scored, f)
    print(f"Wrote {combined_path} n={len(combined)} errors={errors}")


if __name__ == "__main__":
    main()
