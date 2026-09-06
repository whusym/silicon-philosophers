#!/usr/bin/env python3
"""
Evaluate Grok 4.3 via OpenRouter, mirroring 3_model_eval.py behavior.

Same persona + PhilPapers prompts, temperature=0, max_tokens=100,
JSON-list parsing/validation with retries, resume support, and a result
schema compatible with the 4a/4b/4c post-processing scripts.

OpenRouter model: x-ai/grok-4.3

Usage:
    export OPENROUTER_API_KEY="sk-or-..."

    # Smoke test (sync)
    python 3b_eval_openrouter_grok43.py --mode sync --limit 5

    # Full sync run (slow for 277×100; prefer batch)
    python 3b_eval_openrouter_grok43.py --mode sync

    # Recommended: OpenRouter Batch API
    python 3b_eval_openrouter_grok43.py --mode batch submit
    python 3b_eval_openrouter_grok43.py --mode batch status
    python 3b_eval_openrouter_grok43.py --mode batch collect

Optional:
    --data-dir DIR   # philosophers_with_countries.json + question_answer_options.json
    --output-dir DIR
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover

    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(0)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_BATCH_URL = "https://openrouter.ai/api/beta/batches"
MODEL_ID = "x-ai/grok-4.3"
MODEL_LABEL = "grok-4.3"

TEMPERATURE = 0.0
# Grok 4.3 is a reasoning model; keep headroom even with reasoning disabled.
MAX_TOKENS = 256
MAX_RETRIES = 5
SAVE_EVERY = 25
POLL_SECONDS = 30
# Match non-reasoning commercial evals in the paper (disable Grok thinking tokens).
REASONING_ENABLED = False

PHILOSOPHERS_FILE = "philosophers_with_countries.json"
QUESTIONS_FILE = "question_answer_options.json"

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = Path(os.environ.get("DATA_DIR", SCRIPT_DIR / "test_data"))
DEFAULT_OUTPUT_DIR = Path(
    os.environ.get("OUTPUT_DIR", SCRIPT_DIR / f"llm_responses_{MODEL_LABEL}")
)


# ---------------------------------------------------------------------------
# Prompts (verbatim from 3_model_eval.py)
# ---------------------------------------------------------------------------


def build_persona_prompt(persona: Dict[str, Any]) -> str:
    """Build persona prompt from philosopher data"""
    specializations = persona.get("areas_of_specialization", [])
    interests = persona.get("areas_of_interest", [])
    phd = persona.get("phd_institution")
    phd_country = persona.get("phd_country")
    phd_year = persona.get("year_of_phd_degree")
    institution = persona.get("current_institution")
    institution_country = persona.get("current_institution_country")

    # Prefer demographic persona (paper style). If demographics are missing but a
    # name is available, condition on the name so prompts are not identical.
    name = persona.get("name")
    if institution or specializations or interests or phd:
        prompt = "You are a professional philosopher"
        if institution:
            prompt += f" at {institution}"
            if institution_country:
                prompt += f" ({institution_country})"
        prompt += ".\n\n"
    elif name:
        prompt = (
            f"You are {name}, a professional philosopher. "
            "Answer the survey as this philosopher would.\n\n"
        )
    else:
        prompt = "You are a professional philosopher.\n\n"

    if phd or phd_country or phd_year:
        prompt += "Your Educational Background:\n"
        if phd:
            prompt += f"- PhD from {phd}"
            if phd_country:
                prompt += f" ({phd_country})"
            if phd_year:
                prompt += f" in {phd_year}"
            prompt += "\n"
        prompt += "\n"

    if specializations:
        prompt += "Your Areas of Specialization:\n"
        for spec in specializations[:5]:
            prompt += f"- {spec}\n"
        prompt += "\n"

    if interests:
        prompt += "Your Areas of Interest:\n"
        for interest in interests[:8]:
            prompt += f"- {interest}\n"
        prompt += "\n"

    return prompt


def build_question_prompt(question_key: str, answer_options: List[str]) -> str:
    """Build question prompt with options"""
    prompt = """You are answering a survey by the reputable website PhilPapers, which collects
responses across various philosophical domains based on your background. You may select one or multiple options if you deem it appropriate to accurately represent your position.

Please respond with your chosen option(s) as a JSON list. If selecting multiple options, ensure they are logically consistent.

Examples of valid responses:
- ["Accept: physicalism"]
- ["Accept: physicalism", "Reject: non-physicalism"]
- ["Lean towards: physicalism", "Lean against: non-physicalism"]

Given your philosophical profile above, please express your stance on the following question.

Question: """

    prompt += question_key + "\n\nAvailable response options:\n"

    for option in answer_options:
        prompt += f"- {option}\n"

    prompt += "\nRespond ONLY with the JSON list, no other text or explanation.\n"

    return prompt


def build_full_prompt(
    philosopher: Dict[str, Any], question_key: str, options: List[str]
) -> str:
    return build_persona_prompt(philosopher) + build_question_prompt(
        question_key, options
    )


# ---------------------------------------------------------------------------
# Parse / validate (same logic as 3_model_eval.py)
# ---------------------------------------------------------------------------


def normalize_option(option: str) -> str:
    if not option:
        return ""
    normalized = str(option).lower().strip()
    normalized = " ".join(normalized.split())
    normalized = normalized.rstrip(".,!?;")
    while "::" in normalized:
        normalized = normalized.replace("::", ":")
    normalized = normalized.replace(" :", ":").replace(":", ": ").replace(":  ", ": ")
    return " ".join(normalized.split())


def parse_response_list(response_text: str) -> List[str]:
    response_text = (response_text or "").strip()
    if response_text.startswith("```"):
        lines = response_text.split("\n")
        response_text = "\n".join(
            lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
        ).strip()

    json_match = re.search(r"\[.*?\]", response_text, re.DOTALL)
    if json_match:
        try:
            parsed = json.loads(json_match.group())
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if item]
        except json.JSONDecodeError:
            pass

    quoted = re.findall(r'"([^"]+)"', response_text)
    if quoted:
        return quoted
    single = re.findall(r"'([^']+)'", response_text)
    if single:
        return single
    return [response_text] if response_text else []


def validate_response(
    parsed_response: List[str], valid_options: List[str]
) -> Tuple[bool, str]:
    if not parsed_response:
        return False, "Empty response"
    if not isinstance(parsed_response, list):
        return False, f"Response is not a list: {type(parsed_response)}"

    normalized_valid = {normalize_option(opt): opt for opt in valid_options}
    normalized_no_colon = {
        normalize_option(opt).replace(":", ""): opt for opt in valid_options
    }

    invalid_items = []
    for item in parsed_response:
        normalized_item = normalize_option(item)
        normalized_item_no_colon = normalized_item.replace(":", "")
        if (
            item in valid_options
            or normalized_item in normalized_valid
            or normalized_item_no_colon in normalized_no_colon
        ):
            continue
        invalid_items.append(item)

    if invalid_items:
        return False, f"Invalid options: {invalid_items[:3]}"
    return True, "Valid"


def philosopher_snapshot(p: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "name": p.get("name", "Unknown"),
        "areas_of_specialization": p.get("areas_of_specialization", []),
        "areas_of_interest": p.get("areas_of_interest", []),
        "phd_institution": p.get("phd_institution"),
        "phd_country": p.get("phd_country"),
        "year_of_phd_degree": p.get("year_of_phd_degree"),
        "current_institution": p.get("current_institution"),
        "current_institution_country": p.get("current_institution_country"),
    }


def combo_id(phil_name: str, question_key: str) -> str:
    return f"{phil_name}||{question_key}"


# ---------------------------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------------------------


def require_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OR_API_KEY")
    if not key:
        raise SystemExit(
            "Set OPENROUTER_API_KEY (or OR_API_KEY) before running.\n"
            "  export OPENROUTER_API_KEY='sk-or-...'"
        )
    return key


def openrouter_headers(api_key: str) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/whusym/silicon-philosophers",
        "X-Title": "silicon-philosophers-grok43-eval",
    }


def chat_completion(api_key: str, prompt: str) -> str:
    """Single sync chat completion via OpenRouter (OpenAI-compatible)."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Install openai: pip install openai") from exc

    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)
    resp = client.chat.completions.create(
        model=MODEL_ID,
        messages=[{"role": "user", "content": prompt}],
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        extra_body={"reasoning": {"enabled": REASONING_ENABLED}},
    )
    return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_data(data_dir: Path) -> Tuple[List[Dict], Dict[str, List[str]]]:
    phil_path = data_dir / PHILOSOPHERS_FILE
    quest_path = data_dir / QUESTIONS_FILE
    if not phil_path.exists():
        raise SystemExit(f"Missing {phil_path}")
    if not quest_path.exists():
        raise SystemExit(f"Missing {quest_path}")

    with open(phil_path) as f:
        philosophers = json.load(f)
    with open(quest_path) as f:
        questions = json.load(f)

    print(f"Loaded {len(philosophers)} philosophers and {len(questions)} questions")
    return philosophers, questions


def iter_tasks(
    philosophers: List[Dict],
    questions: Dict[str, List[str]],
    completed: set,
    limit: Optional[int],
) -> List[Dict[str, Any]]:
    tasks = []
    for philosopher in philosophers:
        phil_name = philosopher.get("name", "Unknown")
        for qkey, options in questions.items():
            if limit is not None and len(tasks) >= limit:
                return tasks
            cid = combo_id(phil_name, qkey)
            if cid in completed:
                continue
            tasks.append(
                {
                    "philosopher": philosopher,
                    "phil_name": phil_name,
                    "question_key": qkey,
                    "options": options,
                    "combo_id": cid,
                }
            )
    return tasks


# ---------------------------------------------------------------------------
# Sync mode
# ---------------------------------------------------------------------------


def run_sync(data_dir: Path, output_dir: Path, limit: Optional[int]) -> None:
    api_key = require_api_key()
    philosophers, questions = load_data(data_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    resume_path = output_dir / "progress.json"
    combined_path = output_dir / f"{MODEL_LABEL}_combined.json"

    completed: set = set()
    results: List[Dict[str, Any]] = []
    if resume_path.exists():
        with open(resume_path) as f:
            progress = json.load(f)
            completed = set(progress.get("completed", []))
    if combined_path.exists():
        with open(combined_path) as f:
            results = json.load(f)

    tasks = iter_tasks(philosophers, questions, completed, limit)
    print(f"Already completed: {len(completed)}")
    print(f"Remaining this run: {len(tasks)}")
    if not tasks:
        print("Nothing to do.")
        return

    failed = 0
    retries = 0
    batch_buf: List[Dict[str, Any]] = []
    start = time.time()

    for task in tqdm(tasks, desc="Grok 4.3 sync", unit="item"):
        prompt = build_full_prompt(
            task["philosopher"], task["question_key"], task["options"]
        )
        attempts: List[Dict[str, Any]] = []
        total_time = 0.0
        final: Optional[Dict[str, Any]] = None

        for attempt in range(MAX_RETRIES):
            try:
                t0 = time.time()
                raw = chat_completion(api_key, prompt)
                elapsed = time.time() - t0
                total_time += elapsed
                parsed = parse_response_list(raw)
                ok, msg = validate_response(parsed, task["options"])
                attempts.append(
                    {
                        "attempt": attempt + 1,
                        "parsed": parsed,
                        "raw": raw,
                        "valid": ok,
                        "validation_msg": msg,
                        "time": elapsed,
                    }
                )
                if ok:
                    final = {
                        "success": True,
                        "parsed": parsed,
                        "raw": raw,
                        "generation_time": total_time,
                        "attempts": attempt + 1,
                        "all_attempts": attempts,
                    }
                    if attempt > 0:
                        retries += 1
                    break
                if attempt == MAX_RETRIES - 1:
                    final = {
                        "success": False,
                        "error": f"Max retries reached. Last validation: {msg}",
                        "parsed": parsed,
                        "raw": raw,
                        "generation_time": total_time,
                        "attempts": MAX_RETRIES,
                        "all_attempts": attempts,
                    }
            except Exception as exc:  # noqa: BLE001
                detail = f"{type(exc).__name__}: {exc}"
                attempts.append({"attempt": attempt + 1, "error": detail, "time": 0})
                time.sleep(min(2**attempt, 20))
                if attempt == MAX_RETRIES - 1:
                    final = {
                        "success": False,
                        "error": f"Exception after {MAX_RETRIES} attempts: {detail}",
                        "generation_time": total_time,
                        "attempts": MAX_RETRIES,
                        "all_attempts": attempts,
                    }

        assert final is not None
        if not final["success"]:
            failed += 1

        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_ID,
            "philosopher": philosopher_snapshot(task["philosopher"]),
            "question": task["question_key"],
            "response": {
                "parsed": final.get("parsed", []),
                "raw": final.get("raw", ""),
                "success": final["success"],
                "error": final.get("error"),
                "generation_time": final.get("generation_time", 0),
                "attempts": final.get("attempts", 1),
                "all_attempts": final.get("all_attempts", []),
            },
        }
        results.append(entry)
        batch_buf.append(entry)
        completed.add(task["combo_id"])

        if len(batch_buf) >= SAVE_EVERY:
            batch_path = output_dir / f"batch_{int(time.time())}.json"
            with open(batch_path, "w") as f:
                json.dump(batch_buf, f, indent=2)
            batch_buf = []
            with open(resume_path, "w") as f:
                json.dump({"completed": sorted(completed)}, f)
            with open(combined_path, "w") as f:
                json.dump(results, f)

    if batch_buf:
        batch_path = output_dir / f"batch_final_{int(time.time())}.json"
        with open(batch_path, "w") as f:
            json.dump(batch_buf, f, indent=2)
    with open(resume_path, "w") as f:
        json.dump({"completed": sorted(completed)}, f)
    with open(combined_path, "w") as f:
        json.dump(results, f)

    elapsed = time.time() - start
    n = len(tasks)
    print("\n" + "=" * 72)
    print("SYNC COMPLETE")
    print(f"Processed: {n}")
    print(f"Failed: {failed}")
    print(f"Retried: {retries}")
    print(f"Elapsed: {elapsed / 60:.1f} min")
    print(f"Combined: {combined_path}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# Batch mode (OpenRouter Batch API)
# ---------------------------------------------------------------------------


def batch_paths(output_dir: Path) -> Dict[str, Path]:
    batch_dir = output_dir / "openrouter_batch"
    batch_dir.mkdir(parents=True, exist_ok=True)
    return {
        "dir": batch_dir,
        "requests": batch_dir / "batch_requests.jsonl",
        "mapping": batch_dir / "batch_mapping.json",
        "batch_id": batch_dir / "batch_id.json",
        "raw_results": batch_dir / "batch_results.json",
        "combined": output_dir / f"{MODEL_LABEL}_combined.json",
    }


def batch_submit(data_dir: Path, output_dir: Path, limit: Optional[int]) -> None:
    import requests

    api_key = require_api_key()
    philosophers, questions = load_data(data_dir)
    paths = batch_paths(output_dir)

    mapping: Dict[str, Any] = {}
    requests_payload: List[Dict[str, Any]] = []
    count = 0

    with open(paths["requests"], "w") as f:
        for philosopher in philosophers:
            phil_name = philosopher.get("name", "Unknown")
            for qkey, options in questions.items():
                if limit is not None and count >= limit:
                    break
                custom_id = f"req_{count}"
                prompt = build_full_prompt(philosopher, qkey, options)
                mapping[custom_id] = {
                    "philosopher_name": phil_name,
                    "question": qkey,
                    "options": options,
                    "philosopher": philosopher_snapshot(philosopher),
                }
                body = {
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": TEMPERATURE,
                    "max_tokens": MAX_TOKENS,
                    "reasoning": {"enabled": REASONING_ENABLED},
                }
                f.write(
                    json.dumps(
                        {
                            "custom_id": custom_id,
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {**body, "model": MODEL_ID},
                        }
                    )
                    + "\n"
                )
                requests_payload.append({"custom_id": custom_id, "body": body})
                count += 1
            if limit is not None and count >= limit:
                break

    with open(paths["mapping"], "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"Prepared {count} requests")
    print(f"  JSONL audit file: {paths['requests']}")
    print(f"  Mapping: {paths['mapping']}")

    # endpoint + model must appear before requests for OpenRouter stream parsing
    payload_json = (
        '{"endpoint":"/v1/chat/completions","model":'
        + json.dumps(MODEL_ID)
        + ',"requests":'
        + json.dumps(requests_payload, separators=(",", ":"))
        + "}"
    )

    print(f"\nSubmitting batch to OpenRouter ({MODEL_ID})...")
    resp = requests.post(
        OPENROUTER_BATCH_URL,
        headers=openrouter_headers(api_key),
        data=payload_json,
        timeout=600,
    )
    if resp.status_code >= 400:
        raise SystemExit(f"Batch submit failed ({resp.status_code}): {resp.text}")

    created = resp.json()
    batch_id = created.get("id")
    if not batch_id:
        raise SystemExit(f"No batch id in response: {created}")

    with open(paths["batch_id"], "w") as f:
        json.dump(
            {
                "id": batch_id,
                "model": MODEL_ID,
                "n_requests": count,
                "submitted_at": datetime.now(timezone.utc).isoformat(),
                "raw": created,
            },
            f,
            indent=2,
        )

    print(f"Submitted batch id={batch_id}")
    print(f"Saved {paths['batch_id']}")
    print("Next: python 3b_eval_openrouter_grok43.py --mode batch status")


def batch_status(output_dir: Path) -> Dict[str, Any]:
    import requests

    api_key = require_api_key()
    paths = batch_paths(output_dir)
    if not paths["batch_id"].exists():
        raise SystemExit(f"Missing {paths['batch_id']}; run submit first.")

    with open(paths["batch_id"]) as f:
        meta = json.load(f)
    batch_id = meta["id"]

    resp = requests.get(
        f"{OPENROUTER_BATCH_URL}/{batch_id}",
        headers=openrouter_headers(api_key),
        timeout=60,
    )
    if resp.status_code >= 400:
        raise SystemExit(f"Status check failed ({resp.status_code}): {resp.text}")
    batch = resp.json()
    status = batch.get("status")
    print(f"Batch {batch_id}: status={status}")
    for k in ("completed", "failed", "total", "request_counts"):
        if k in batch:
            print(f"  {k}: {batch[k]}")
    return batch


def _extract_content(item: Dict[str, Any]) -> Tuple[Optional[str], Any]:
    """Pull assistant text out of varied OpenRouter/OpenAI batch result shapes."""
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


def batch_collect(output_dir: Path) -> None:
    import requests

    api_key = require_api_key()
    paths = batch_paths(output_dir)
    if not paths["batch_id"].exists():
        raise SystemExit(f"Missing {paths['batch_id']}; run submit first.")
    if not paths["mapping"].exists():
        raise SystemExit(f"Missing {paths['mapping']}")

    with open(paths["batch_id"]) as f:
        meta = json.load(f)
    with open(paths["mapping"]) as f:
        mapping = json.load(f)

    batch_id = meta["id"]
    print(f"Polling batch {batch_id} until terminal...")
    while True:
        resp = requests.get(
            f"{OPENROUTER_BATCH_URL}/{batch_id}",
            headers=openrouter_headers(api_key),
            timeout=120,
        )
        if resp.status_code >= 400:
            raise SystemExit(f"Poll failed ({resp.status_code}): {resp.text}")
        batch = resp.json()
        status = batch.get("status")
        print(f"  status={status}")
        if status in {"completed", "failed", "cancelled", "expired", "ended"}:
            break
        time.sleep(POLL_SECONDS)

    with open(paths["raw_results"], "w") as f:
        json.dump(batch, f, indent=2)
    print(f"Saved raw batch payload → {paths['raw_results']}")

    raw_results = (
        batch.get("results") or batch.get("output") or batch.get("data") or []
    )
    if isinstance(raw_results, dict):
        raw_results = raw_results.get("results") or raw_results.get("data") or []

    if not raw_results:
        output_url = batch.get("output_file_url") or batch.get("results_url")
        if output_url:
            r = requests.get(
                output_url, headers=openrouter_headers(api_key), timeout=300
            )
            r.raise_for_status()
            text = r.text.strip()
            if text.startswith("["):
                raw_results = json.loads(text)
            else:
                raw_results = [
                    json.loads(line) for line in text.splitlines() if line.strip()
                ]

    if not raw_results:
        raise SystemExit(
            "Batch finished but no results found in response. "
            f"Inspect {paths['raw_results']} and adjust collect parsing if needed."
        )

    combined: List[Dict[str, Any]] = []
    success = 0
    failed = 0

    for item in raw_results:
        custom_id = item.get("custom_id") or item.get("customId")
        meta_row = mapping.get(custom_id)
        if not meta_row:
            failed += 1
            continue

        content, err = _extract_content(item)
        parsed = parse_response_list(content or "") if content else []
        ok, msg = (
            validate_response(parsed, meta_row["options"])
            if content
            else (False, "no content")
        )
        if ok:
            success += 1
        else:
            failed += 1
            if not err:
                err = msg

        combined.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "model": MODEL_ID,
                "philosopher": meta_row["philosopher"],
                "question": meta_row["question"],
                "response": {
                    "parsed": parsed,
                    "raw": content or "",
                    "success": bool(ok),
                    "error": None
                    if ok
                    else (err if isinstance(err, str) else json.dumps(err)),
                    "generation_time": None,
                    "attempts": 1,
                    "all_attempts": [],
                    "custom_id": custom_id,
                },
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(paths["combined"], "w") as f:
        json.dump(combined, f, indent=2)

    print("\n" + "=" * 72)
    print("BATCH COLLECT COMPLETE")
    print(f"Results: {len(combined)}  success≈{success}  failed/invalid≈{failed}")
    print(f"Combined: {paths['combined']}")
    print("Next: feed this into 4a_process_model_results.py / 4b_merge_llm_responses.py")
    print("=" * 72)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OpenRouter Grok 4.3 eval (mirrors 3_model_eval.py)"
    )
    p.add_argument(
        "--mode",
        choices=["sync", "batch"],
        default="sync",
        help="sync = live API calls; batch = OpenRouter Batch API",
    )
    p.add_argument(
        "batch_action",
        nargs="?",
        choices=["submit", "status", "collect"],
        help="Required when --mode batch",
    )
    p.add_argument(
        "--limit", "-l", type=int, default=None, help="Limit # of items (testing)"
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Dir with {PHILOSOPHERS_FILE} + {QUESTIONS_FILE}",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Where to write responses / progress",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("=" * 72)
    print("OpenRouter Grok 4.3 evaluation")
    print(f"Model: {MODEL_ID}")
    print(f"Mode: {args.mode}")
    print(f"Data: {args.data_dir}")
    print(f"Output: {args.output_dir}")
    if args.limit:
        print(f"Limit: {args.limit}")
    print("=" * 72)

    if args.mode == "sync":
        run_sync(args.data_dir, args.output_dir, args.limit)
        return

    if not args.batch_action:
        raise SystemExit("With --mode batch, pass: submit | status | collect")
    if args.batch_action == "submit":
        batch_submit(args.data_dir, args.output_dir, args.limit)
    elif args.batch_action == "status":
        batch_status(args.output_dir)
    elif args.batch_action == "collect":
        batch_collect(args.output_dir)


if __name__ == "__main__":
    main()
