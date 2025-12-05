#!/usr/bin/env python3
"""
Process OpenAI batch results and merge with philosopher data.
Converts OpenAI batch output format to match combined.json structure.
"""

import json
import re
import os
import glob
from datetime import datetime
from collections import defaultdict

# Configuration (use relative paths from script directory)
OPENAI_RESULTS_DIR = './openai_results'
MAPPING_FILE = f'{OPENAI_RESULTS_DIR}/openai_batch_input_mapping.json'
PHILOSOPHERS_FILE = './philosophers_with_countries.json'
OUTPUT_COMBINED = './result_analysis/openai_gpt4o_combined.json'
OUTPUT_MERGED = './merged_openai_gpt4o_philosophers.json'

MODEL_NAME = "gpt-4o-2024-08-06"

def extract_json_from_content(content):
    """Extract JSON array from markdown code block or plain JSON."""
    content = content.strip()
    
    # Try to extract from markdown code block
    json_match = re.search(r'```(?:json)?\s*(\[.*?\])\s*```', content, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # Try parsing as plain JSON
    try:
        parsed = json.loads(content)
        if isinstance(parsed, list):
            return parsed
        elif isinstance(parsed, str):
            # Sometimes it's a string representation of a list
            return json.loads(parsed)
    except json.JSONDecodeError:
        pass
    
    # If all else fails, try to extract array-like content
    array_match = re.search(r'\[.*?\]', content, re.DOTALL)
    if array_match:
        try:
            return json.loads(array_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # Return as single-item list if it's a plain string
    return [content]

def parse_response_text(response_text):
    """Parse response text into stance and answer."""
    response_text = response_text.strip()
    
    # Handle Agnostic/undecided
    if response_text.lower() == 'agnostic/undecided':
        return 'Agnostic/undecided', 'agnostic/undecided'
    
    # Patterns to match
    patterns = [
        (r'^Accept:\s*(.+)$', 'Accept'),
        (r'^Accept an alternative view:\s*(.+)$', 'Accept'),
        (r'^Accept an alternative:\s*(.+)$', 'Accept'),
        (r'^Lean towards:\s*(.+)$', 'Lean towards'),
        (r'^Lean toward:\s*(.+)$', 'Lean towards'),
        (r'^Neutral towards:\s*(.+)$', 'Neutral'),
        (r'^Neutral toward:\s*(.+)$', 'Neutral'),
        (r'^Neutral:\s*(.+)$', 'Neutral'),
        (r'^Lean against:\s*(.+)$', 'Lean against'),
        (r'^Reject:\s*(.+)$', 'Reject'),
    ]
    
    for pattern, stance in patterns:
        match = re.match(pattern, response_text, re.IGNORECASE)
        if match:
            answer = match.group(1).strip()
            return stance, answer
    
    # If no pattern matches, return as-is
    return None, response_text

def process_openai_batch_results():
    """Process all OpenAI batch output files and convert to combined format."""
    print("Loading input mapping...")
    with open(MAPPING_FILE, 'r') as f:
        input_mapping = json.load(f)
    
    print(f"Loaded mapping for {len(input_mapping)} requests")
    
    # Find all batch output files
    output_files = sorted(glob.glob(f'{OPENAI_RESULTS_DIR}/openai_batch_output_*.jsonl'))
    print(f"Found {len(output_files)} batch output files")
    
    combined_results = []
    processed_count = 0
    error_count = 0
    
    for output_file in output_files:
        print(f"Processing {output_file}...")
        with open(output_file, 'r') as f:
            for line_num, line in enumerate(f):
                if not line.strip():
                    continue
                
                try:
                    batch_entry = json.loads(line)
                    custom_id = batch_entry.get('custom_id')
                    
                    if custom_id not in input_mapping:
                        print(f"Warning: custom_id {custom_id} not found in mapping")
                        error_count += 1
                        continue
                    
                    mapping = input_mapping[custom_id]
                    philosopher = mapping['philosopher']
                    question = mapping['question']
                    
                    # Extract response from batch entry
                    response_obj = batch_entry.get('response', {})
                    
                    if response_obj.get('status_code') != 200:
                        print(f"Warning: Non-200 status for {custom_id}: {response_obj.get('status_code')}")
                        error_count += 1
                        continue
                    
                    body = response_obj.get('body', {})
                    choices = body.get('choices', [])
                    
                    if not choices:
                        print(f"Warning: No choices for {custom_id}")
                        error_count += 1
                        continue
                    
                    message = choices[0].get('message', {})
                    content = message.get('content', '')
                    
                    if not content:
                        print(f"Warning: Empty content for {custom_id}")
                        error_count += 1
                        continue
                    
                    # Extract JSON array from content
                    parsed_responses = extract_json_from_content(content)
                    
                    if not parsed_responses:
                        print(f"Warning: Could not parse response for {custom_id}")
                        error_count += 1
                        continue
                    
                    # Process parsed responses
                    processed_parsed = []
                    for resp_text in parsed_responses:
                        stance, answer = parse_response_text(resp_text)
                        if stance:
                            processed_parsed.append(f"{stance}: {answer}")
                        else:
                            processed_parsed.append(resp_text)
                    
                    # Get usage info
                    usage = body.get('usage', {})
                    generation_time = 0  # OpenAI batch doesn't provide timing
                    
                    # Create entry matching combined.json structure
                    entry = {
                        "timestamp": datetime.now().isoformat(),
                        "model": MODEL_NAME,
                        "philosopher": philosopher,
                        "question": question,
                        "response": {
                            "parsed": processed_parsed,
                            "raw": json.dumps(parsed_responses),
                            "success": True,
                            "error": None,
                            "generation_time": generation_time,
                            "attempts": 1,
                            "all_attempts": [
                                {
                                    "attempt": 1,
                                    "parsed": processed_parsed,
                                    "raw": json.dumps(parsed_responses),
                                    "valid": True,
                                    "validation_msg": "Valid",
                                    "time": generation_time,
                                    "usage": {
                                        "input_tokens": usage.get('prompt_tokens', 0),
                                        "output_tokens": usage.get('completion_tokens', 0)
                                    }
                                }
                            ]
                        }
                    }
                    
                    combined_results.append(entry)
                    processed_count += 1
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON on line {line_num} of {output_file}: {e}")
                    error_count += 1
                except Exception as e:
                    print(f"Error processing entry: {e}")
                    error_count += 1
    
    print(f"\nProcessed {processed_count} entries")
    print(f"Errors: {error_count}")
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(OUTPUT_COMBINED), exist_ok=True)
    
    # Save combined results
    print(f"\nSaving combined results to {OUTPUT_COMBINED}...")
    with open(OUTPUT_COMBINED, 'w') as f:
        json.dump(combined_results, f, indent=2)
    
    print(f"Saved {len(combined_results)} entries to combined file")
    
    return combined_results

def merge_with_philosophers(combined_results):
    """Merge combined results with philosopher data."""
    # Import merge function
    try:
        from merge_llm_responses_with_philosophers import merge_llm_responses
        print(f"\nMerging with philosopher data...")
        merge_llm_responses(OUTPUT_COMBINED, OUTPUT_MERGED, PHILOSOPHERS_FILE)
    except ImportError:
        print("\nNote: merge_llm_responses_with_philosophers.py not found.")
        print("Run 4b_merge_llm_responses.py separately to merge results.")

def main():
    print("="*80)
    print("Processing OpenAI Batch Results")
    print("="*80)
    
    # Process batch results
    combined_results = process_openai_batch_results()
    
    # Merge with philosophers
    merge_with_philosophers(combined_results)
    
    print("\n" + "="*80)
    print("✅ Processing complete!")
    print("="*80)
    print(f"Combined file: {OUTPUT_COMBINED}")
    print(f"Merged file: {OUTPUT_MERGED}")

if __name__ == '__main__':
    main()
