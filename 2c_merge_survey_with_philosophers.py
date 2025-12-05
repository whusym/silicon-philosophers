#!/usr/bin/env python3
"""
Merge normalized human survey responses with philosopher profile data.

This creates the merged_human_survey_philosophers.json file used for 
comparing human responses with LLM responses.

Requires:
- normalized_survey_responses.json (human survey responses with normalized scores)
- philosophers_with_countries.json (philosopher profiles with demographics)

Outputs:
- merged_human_survey_philosophers.json (human data in same format as LLM merged files)
"""

import json
import os

# Configuration - update these paths as needed
BASE_DIR = "."  # Working directory
SURVEY_RESPONSES_FILE = "normalized_survey_responses.json"
PHILOSOPHERS_FILE = "philosophers_with_countries.json"
OUTPUT_FILE = "merged_human_survey_philosophers.json"


def extract_slug_from_url(url):
    """Extract slug from philpeople.org URL."""
    if url and 'philpeople.org/profiles/' in url:
        return url.split('philpeople.org/profiles/')[-1]
    return None


def main():
    # Load both JSON files
    print(f"Loading {SURVEY_RESPONSES_FILE}...")
    survey_path = os.path.join(BASE_DIR, SURVEY_RESPONSES_FILE)
    with open(survey_path, 'r') as f:
        survey_responses = json.load(f)
    
    print(f"Loading {PHILOSOPHERS_FILE}...")
    philosophers_path = os.path.join(BASE_DIR, PHILOSOPHERS_FILE)
    with open(philosophers_path, 'r') as f:
        philosophers = json.load(f)
    
    print(f"Found {len(survey_responses)} survey response entries")
    print(f"Found {len(philosophers)} philosopher entries")
    
    # Create a mapping from slug to philosopher data
    philosopher_by_slug = {}
    for philosopher in philosophers:
        slug = extract_slug_from_url(philosopher.get('url'))
        if slug:
            philosopher_by_slug[slug] = philosopher
    
    print(f"Created slug mapping for {len(philosopher_by_slug)} philosophers")
    
    # Merge the data
    merged_results = []
    matched_count = 0
    unmatched_count = 0
    
    for survey_entry in survey_responses:
        philosopher_key = survey_entry.get('philosopher')
        
        if philosopher_key in philosopher_by_slug:
            # Merge: start with philosopher data, add survey responses
            merged_entry = philosopher_by_slug[philosopher_key].copy()
            merged_entry['responses'] = survey_entry.get('responses', {})
            merged_results.append(merged_entry)
            matched_count += 1
        else:
            # Keep survey entry even if no match (with a note)
            merged_entry = survey_entry.copy()
            merged_entry['matched'] = False
            merged_results.append(merged_entry)
            unmatched_count += 1
            if unmatched_count <= 5:  # Only print first 5 warnings
                print(f"Warning: No match found for philosopher '{philosopher_key}'")
    
    if unmatched_count > 5:
        print(f"... and {unmatched_count - 5} more unmatched")
    
    # Save merged results
    output_path = os.path.join(BASE_DIR, OUTPUT_FILE)
    print(f"\nSaving merged results to {output_path}...")
    with open(output_path, 'w') as f:
        json.dump(merged_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✅ Merge complete!")
    print(f"{'='*60}")
    print(f"   - Total entries: {len(merged_results)}")
    print(f"   - Matched: {matched_count}")
    print(f"   - Unmatched: {unmatched_count}")
    
    # Show some sample merged entries
    print(f"\n📋 Sample merged entries:")
    for i, entry in enumerate(merged_results[:3], 1):
        print(f"\n{i}. {entry.get('name', entry.get('philosopher', 'Unknown'))}")
        if 'responses' in entry:
            response_count = len(entry['responses'])
            print(f"   Responses: {response_count} survey responses")
        if 'phd_country' in entry:
            print(f"   PhD Country: {entry.get('phd_country')}")


if __name__ == '__main__':
    main()

