#!/usr/bin/env python3
"""
Normalize question names across human and LLM datasets.

This is critical for enabling KL divergence computation between
human and LLM response distributions. Without normalization,
questions like "A priori knowledge: yes" won't match "a priori knowledge: yes".

Key normalizations:
- Lowercase first letter
- Ensure space after colon
- Strip whitespace

Requires:
- merged_*_philosophers.json files

Outputs:
- merged_*_philosophers_normalized.json files
"""

import json
import re
import os

# Configuration - add your model files here
MODEL_FILES = {
    'human': 'merged_human_survey_philosophers.json',
    'llama3p1_8b': 'merged_llama3p18b_philosophers.json',
    'mistral7b': 'merged_mistral7b_philosophers.json',
    'gpt4o': 'merged_openai_gpt4o_philosophers.json',
    'qwen3_4b': 'merged_qwen3-4b_philosophers.json',
    'sonnet45': 'merged_sonnet45_philosophers.json',
}


def normalize_question_name(q):
    """
    Normalize question name to common format:
    - Lowercase first letter
    - Ensure space after colon
    - Strip whitespace
    """
    q = q.strip()

    # Add space after colon if missing
    q = re.sub(r':([^ ])', r': \1', q)

    # Lowercase the first character
    if len(q) > 0:
        q = q[0].lower() + q[1:]

    return q


def normalize_json_file(input_path, output_path):
    """Load JSON, normalize question names, save"""
    with open(input_path, 'r') as f:
        data = json.load(f)

    # Normalize all question names in responses
    for phil in data:
        if 'responses' in phil:
            old_responses = phil['responses']
            new_responses = {}

            for q, v in old_responses.items():
                normalized_q = normalize_question_name(q)
                new_responses[normalized_q] = v

            phil['responses'] = new_responses

    # Save normalized version
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Normalized: {input_path} → {output_path}")
    return data


def main():
    print("="*80)
    print("NORMALIZING QUESTION NAMES")
    print("="*80)
    print("\nThis enables KL divergence computation between human and LLM data.")

    base_dir = "."  # Current directory
    
    print("\nNormalizing all files...")
    print()
    
    for model, filename in MODEL_FILES.items():
        input_path = os.path.join(base_dir, filename)
        
        if not os.path.exists(input_path):
            print(f"  {model}: ⚠️  File not found: {filename}")
            continue
            
        output_path = input_path.replace('.json', '_normalized.json')

        # Show sample before/after
        with open(input_path, 'r') as f:
            original_data = json.load(f)

        original_questions = list(original_data[0].get('responses', {}).keys())[:3]

        # Normalize
        normalized_data = normalize_json_file(input_path, output_path)

        normalized_questions = list(normalized_data[0].get('responses', {}).keys())[:3]

        print(f"\n  {model}:")
        print(f"    Before: {original_questions[0] if original_questions else 'N/A'}")
        print(f"    After:  {normalized_questions[0] if normalized_questions else 'N/A'}")

    print("\n" + "="*80)
    print("NORMALIZATION COMPLETE")
    print("="*80)
    print("\nNormalized files saved with '_normalized.json' suffix")
    print("\nNext: Run 5a_compute_quality_metrics.py using the normalized files")


if __name__ == "__main__":
    main()

