#!/usr/bin/env python3
"""
Merge LLM responses with philosopher profile data.

This script converts raw LLM response JSONs (from model evaluation) into
the same format as human survey data, enabling direct comparison.

Key features:
- Parses "Stance: answer" format responses
- Converts stances to numeric scores (0-1 scale)
- Matches LLM responses with philosopher profiles by name

Requires:
- LLM combined JSON file (e.g., result_analysis/llama3p18b_combined.json)
- philosophers_with_countries.json (philosopher profiles)

Outputs:
- merged_*_philosophers.json (e.g., merged_llama3p18b_philosophers.json)
"""

import json
import re
from collections import defaultdict
from urllib.parse import urlparse

# Configuration
PHILOSOPHERS_FILE = "philosophers_with_countries.json"

# Stance to score mapping (0-1 scale)
STANCE_SCORES = {
    'Accept': 1.0,
    'Lean towards': 0.75,
    'Lean toward': 0.75,
    'Neutral towards': 0.5,
    'Neutral toward': 0.5,
    'Neutral': 0.5,
    'Lean against': 0.25,
    'Reject': 0.0,
    'Agnostic/undecided': 0.5,
}

# Known philosophical opposites (for reference)
PHILOSOPHICAL_OPPOSITES = {
    'Platonism': 'nominalism',
    'nominalism': 'Platonism',
    'physicalism': 'dualism',
    'dualism': 'physicalism',
    'internalism': 'externalism',
    'externalism': 'internalism',
    'compatibilism': 'incompatibilism',
    'incompatibilism': 'compatibilism',
    'theism': 'atheism',
    'atheism': 'theism',
    'moral realism': 'moral anti-realism',
    'moral anti-realism': 'moral realism',
    'consequentialism': 'deontology',
    'deontology': 'consequentialism',
    'permissible': 'impermissible',
    'impermissible': 'permissible',
}


def extract_slug_from_url(url):
    """Extract slug from philpeople.org URL."""
    if url and 'philpeople.org/profiles/' in url:
        return url.split('philpeople.org/profiles/')[-1]
    return None


def normalize_name(name):
    """Normalize philosopher name for matching."""
    if not name:
        return None
    normalized = re.sub(r'\s+', ' ', name.strip().lower())
    return normalized


def parse_llm_response(response_text):
    """Parse "Stance: answer" format into (stance, answer)."""
    stances = [
        ('Accept:', 'Accept'),
        ('Accept an alternative view:', 'Accept'),
        ('Accept an alternative:', 'Accept'),
        ('Lean towards:', 'Lean towards'),
        ('Lean toward:', 'Lean toward'),
        ('Neutral towards:', 'Neutral'),
        ('Neutral toward:', 'Neutral'),
        ('Neutral:', 'Neutral'),
        ('Lean against:', 'Lean against'),
        ('Reject:', 'Reject'),
        ('Agnostic/undecided', 'Agnostic/undecided'),
    ]
    
    response_text = response_text.strip()
    
    # Check for Agnostic/undecided first (no colon)
    if response_text.lower() == 'agnostic/undecided':
        return 'Agnostic/undecided', 'agnostic/undecided'
    
    for stance_text, stance_name in stances:
        if response_text.lower().startswith(stance_text.lower()):
            answer = response_text[len(stance_text):].strip()
            return stance_name, answer
    
    return None, response_text


def build_philosopher_responses(llm_data):
    """
    Build philosopher responses dictionary from LLM data.
    Returns: {philosopher_name: {question:answer: score}}
    """
    philosopher_responses = defaultdict(dict)
    
    print("Processing LLM responses...")
    
    for entry in llm_data:
        philosopher_data = entry.get('philosopher', {})
        philosopher_name = philosopher_data.get('name')
        question = entry.get('question', '')
        response = entry.get('response', {})
        
        if not philosopher_name or not question or not response.get('success', False):
            continue
        
        # Parse responses
        parsed_responses = response.get('parsed', [])
        if isinstance(parsed_responses, str):
            parsed_responses = [parsed_responses]
        
        # Process each parsed response
        for resp_text in parsed_responses:
            stance, answer = parse_llm_response(resp_text)
            
            if stance is None:
                continue
            
            # Handle Agnostic/undecided
            if stance == 'Agnostic/undecided':
                qa_key = f"{question}:agnostic/undecided"
                score = STANCE_SCORES.get(stance, 0.5)
            else:
                qa_key = f"{question}:{answer}"
                score = STANCE_SCORES.get(stance)
                if score is None:
                    continue
            
            # Store response (keep first valid response for each question:answer pair)
            if qa_key not in philosopher_responses[philosopher_name]:
                philosopher_responses[philosopher_name][qa_key] = score
    
    print(f"Processed responses for {len(philosopher_responses)} philosophers")
    return philosopher_responses


def merge_llm_responses(input_file, output_file, philosophers_file=PHILOSOPHERS_FILE):
    """Merge LLM responses with philosopher data."""
    # Load LLM combined data
    print(f"Loading LLM responses from {input_file}...")
    with open(input_file, 'r') as f:
        llm_data = json.load(f)
    
    print(f"Loaded {len(llm_data)} LLM response entries")
    
    # Load philosopher data
    print(f"Loading philosopher data from {philosophers_file}...")
    with open(philosophers_file, 'r') as f:
        philosophers = json.load(f)
    
    print(f"Loaded {len(philosophers)} philosopher entries")
    
    # Build philosopher responses from LLM data
    llm_responses = build_philosopher_responses(llm_data)
    
    # Create mappings for matching
    philosopher_by_slug = {}
    philosopher_by_name = {}
    
    for philosopher in philosophers:
        slug = extract_slug_from_url(philosopher.get('url'))
        if slug:
            philosopher_by_slug[slug] = philosopher
        
        name = philosopher.get('name')
        if name:
            normalized_name = normalize_name(name)
            if normalized_name:
                philosopher_by_name[normalized_name] = philosopher
    
    print(f"Created slug mapping for {len(philosopher_by_slug)} philosophers")
    print(f"Created name mapping for {len(philosopher_by_name)} philosophers")
    
    # Merge the data
    merged_results = []
    matched_count = 0
    unmatched_count = 0
    
    for philosopher_name, responses in llm_responses.items():
        # Try to find matching philosopher
        matched_philosopher = None
        
        # Try by normalized name first
        normalized_name = normalize_name(philosopher_name)
        if normalized_name and normalized_name in philosopher_by_name:
            matched_philosopher = philosopher_by_name[normalized_name]
        else:
            # Try exact name match
            for phil in philosophers:
                if normalize_name(phil.get('name')) == normalized_name:
                    matched_philosopher = phil
                    break
        
        if matched_philosopher:
            # Merge: start with philosopher data, add LLM responses
            merged_entry = matched_philosopher.copy()
            merged_entry['responses'] = responses
            merged_results.append(merged_entry)
            matched_count += 1
        else:
            # Create entry from LLM data even if no match
            merged_entry = {
                'name': philosopher_name,
                'responses': responses,
                'matched': False,
                'source': 'llm'
            }
            # Try to get philosopher info from LLM data
            for entry in llm_data:
                phil_data = entry.get('philosopher', {})
                if phil_data.get('name') == philosopher_name:
                    merged_entry.update({
                        'areas_of_specialization': phil_data.get('areas_of_specialization', []),
                        'areas_of_interest': phil_data.get('areas_of_interest', []),
                        'phd_institution': phil_data.get('phd_institution'),
                        'phd_country': phil_data.get('phd_country'),
                        'year_of_phd_degree': phil_data.get('year_of_phd_degree'),
                        'current_institution': phil_data.get('current_institution'),
                        'current_institution_country': phil_data.get('current_institution_country'),
                    })
                    break
            
            merged_results.append(merged_entry)
            unmatched_count += 1
            if unmatched_count <= 10:
                print(f"Warning: No match found for philosopher '{philosopher_name}'")
    
    # Save merged results
    print(f"\nSaving merged results to {output_file}...")
    with open(output_file, 'w') as f:
        json.dump(merged_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Merge complete!")
    print(f"{'='*60}")
    print(f"   - Total entries: {len(merged_results)}")
    print(f"   - Matched: {matched_count}")
    print(f"   - Unmatched: {unmatched_count}")
    
    # Show sample
    print(f"\n📋 Sample merged entries:")
    for i, entry in enumerate(merged_results[:3], 1):
        name = entry.get('name', 'Unknown')
        response_count = len(entry.get('responses', {}))
        print(f"\n{i}. {name}")
        print(f"   Responses: {response_count} LLM responses")
        if 'phd_country' in entry:
            print(f"   PhD Country: {entry.get('phd_country')}")
        if 'matched' in entry and not entry['matched']:
            print(f"   ⚠️  Not matched with philosopher database")


def main():
    """
    Example usage: Process multiple model outputs.
    Modify the files_to_process list for your use case.
    """
    # Configuration - adjust paths as needed
    base_dir = '.'
    philosophers_file = f'{base_dir}/philosophers_with_countries.json'
    
    # Files to process (modify as needed)
    files_to_process = [
        {
            'input': f'{base_dir}/result_analysis/llama3p18b_combined.json',
            'output': f'{base_dir}/merged_llama3p18b_philosophers.json'
        },
        {
            'input': f'{base_dir}/result_analysis/mistral7b_combined.json',
            'output': f'{base_dir}/merged_mistral7b_philosophers.json'
        },
        {
            'input': f'{base_dir}/result_analysis/qwen3-4b_combined.json',
            'output': f'{base_dir}/merged_qwen3-4b_philosophers.json'
        },
        {
            'input': f'{base_dir}/result_analysis/sonnet45_combined.json',
            'output': f'{base_dir}/merged_sonnet45_philosophers.json'
        },
    ]
    
    import os
    for file_info in files_to_process:
        if not os.path.exists(file_info['input']):
            print(f"Skipping {file_info['input']} (file not found)")
            continue
            
        print(f"\n{'='*80}")
        print(f"Processing: {file_info['input']}")
        print(f"{'='*80}")
        merge_llm_responses(file_info['input'], file_info['output'], philosophers_file)
        print()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) == 3:
        # Usage: python 4b_merge_llm_responses.py <input_file> <output_file>
        input_file = sys.argv[1]
        output_file = sys.argv[2]
        merge_llm_responses(input_file, output_file)
    else:
        # Run default batch processing
        main()

