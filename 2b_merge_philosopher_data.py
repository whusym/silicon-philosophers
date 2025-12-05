#!/usr/bin/env python3
"""
Merge philosopher_survey_responses.json with philosophers_with_countries.json

This script merges:
- Survey responses extracted from views HTML files
- Philosopher profile data with country information

Requires:
- philosopher_survey_responses.json (from 2_process_crawled_html.py)
- philosophers_with_countries.json (philosopher profiles with country info)

Outputs:
- philosophers_merged.json
"""

import json
from pathlib import Path
from urllib.parse import urlparse

# Configuration
SURVEY_RESPONSES_FILE = "philosopher_survey_responses.json"
PHILOSOPHERS_FILE = "philosophers_with_countries.json"
OUTPUT_FILE = "philosophers_merged.json"


def extract_slug_from_url(url):
    """Extract philosopher slug from URL like 'https://philpeople.org/profiles/sean-aas' -> 'sean-aas'"""
    if not url:
        return None
    path = urlparse(url).path
    parts = path.strip('/').split('/')
    if len(parts) >= 2 and parts[-2] == 'profiles':
        return parts[-1]
    return None


def merge_philosopher_data():
    """Merge survey responses with philosopher country data"""
    
    # Load survey responses
    print(f"Loading {SURVEY_RESPONSES_FILE}...")
    with open(SURVEY_RESPONSES_FILE, 'r', encoding='utf-8') as f:
        survey_responses = json.load(f)
    
    # Create a dictionary keyed by philosopher slug for quick lookup
    survey_dict = {item['philosopher']: item['responses'] for item in survey_responses}
    print(f"Loaded {len(survey_dict)} philosophers with survey responses")
    
    # Load philosopher country data
    print(f"Loading {PHILOSOPHERS_FILE}...")
    with open(PHILOSOPHERS_FILE, 'r', encoding='utf-8') as f:
        philosophers = json.load(f)
    
    print(f"Loaded {len(philosophers)} philosophers with country data")
    
    # Merge the data
    merged_results = []
    matched_count = 0
    unmatched_survey = []
    
    # First, match philosophers from philosophers_with_countries.json with survey responses
    for philosopher in philosophers:
        url = philosopher.get('url', '')
        slug = extract_slug_from_url(url)
        
        merged_entry = philosopher.copy()
        
        if slug and slug in survey_dict:
            merged_entry['responses'] = survey_dict[slug]
            matched_count += 1
        else:
            # No survey responses found for this philosopher
            merged_entry['responses'] = {}
        
        merged_results.append(merged_entry)
    
    # Find survey responses that don't have matching philosopher entries
    matched_slugs = {extract_slug_from_url(p.get('url', '')) for p in philosophers}
    for slug, responses in survey_dict.items():
        if slug not in matched_slugs:
            unmatched_survey.append({
                'philosopher': slug,
                'responses': responses
            })
    
    # Add unmatched survey responses to the results
    if unmatched_survey:
        print(f"\nFound {len(unmatched_survey)} philosophers with survey responses but no country data")
        for entry in unmatched_survey:
            merged_results.append({
                'philosopher': entry['philosopher'],
                'responses': entry['responses'],
                'name': None,
                'url': None,
                'areas_of_specialization': [],
                'areas_of_interest': [],
                'phd_institution': None,
                'current_institution': None,
                'survey_response_count': len(entry['responses']),
                'phd_country': None,
                'current_institution_country': None,
                'year_of_phd_degree': None
            })
    
    # Write merged results
    print(f"\nWriting merged data to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"✅ Merge complete!")
    print(f"{'='*60}")
    print(f"Total entries: {len(merged_results)}")
    print(f"Matched philosophers: {matched_count}")
    print(f"Philosophers with survey responses only: {len(unmatched_survey)}")
    print(f"Results written to {OUTPUT_FILE}")


if __name__ == '__main__':
    merge_philosopher_data()

