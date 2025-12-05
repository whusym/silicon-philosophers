#!/usr/bin/env python3
"""
Scrape detailed profile information from PhilPeople profile pages.

This script extracts:
- Areas of Specialization
- Areas of Interest
- PhD Institution
- Current Institution
- Location

Requires: philpeople_profiles.json (list of profile URLs)
Outputs: philosopher_details.json
"""

import requests
from bs4 import BeautifulSoup
import time
import json
import os
from typing import Dict, List

# Configuration
INPUT_FILE = "philpeople_profiles.json"  # List of profile URLs
OUTPUT_FILE = "philosopher_details.json"
REQUEST_DELAY = 2  # Seconds between requests (be polite!)
SAVE_INTERVAL = 10  # Save progress every N profiles


def scrape_profile_details(profile_url: str) -> Dict:
    """
    Scrape detailed information from a PhilPeople profile page.
    
    Returns:
        Dictionary containing:
        - areas_of_specialization: list of areas
        - areas_of_interest: list of areas
        - location: string (if available)
        - phd_institution: string (if available)
        - current_institution: string (if available)
    """
    details = {
        'profile_url': profile_url,
        'areas_of_specialization': [],
        'areas_of_interest': [],
        'location': None,
        'phd_institution': None,
        'current_institution': None,
        'error': None
    }
    
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        response = requests.get(profile_url, headers=headers, timeout=15)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Extract current institution from the header
        affil_div = soup.find('div', class_='affil')
        if affil_div:
            details['current_institution'] = affil_div.get_text(strip=True)
        
        # Find Areas of Specialization and Interest
        about_topics = soup.find_all('div', class_='about__topics')
        for section in about_topics:
            content_div = section.find('div', class_='about__content')
            if not content_div:
                continue
            
            title_span = content_div.find('span', class_='about-title')
            if not title_span:
                continue
            
            title_text = title_span.get_text(strip=True)
            
            # Find all category links in the table
            table = content_div.find('table')
            if table:
                category_links = table.find_all('a', class_='category')
                areas = [link.get_text(strip=True) for link in category_links]
                
                if 'Areas of Specialization' in title_text:
                    details['areas_of_specialization'] = areas
                elif 'Areas of Interest' in title_text:
                    details['areas_of_interest'] = areas
        
        # Find PhD info from "row about" sections
        about_sections = soup.find_all('div', class_='row about')
        
        for section in about_sections:
            content_div = section.find('div', class_='about__content')
            if not content_div:
                continue
            
            title_span = content_div.find('span', class_='about-title')
            if not title_span:
                continue
            
            # Check if this section has degree information
            extra_div = content_div.find('div', class_='text-muted about-extra')
            if extra_div:
                extra_text = extra_div.get_text(strip=True)
                # Check if it mentions PhD
                if 'PhD' in extra_text or 'Ph.D' in extra_text:
                    institution_link = title_span.find('a', class_='affil')
                    if institution_link:
                        institution_name = institution_link.get_text(strip=True)
                        
                        second_div = content_div.find('div', class_='about-second')
                        if second_div:
                            department = second_div.get_text(strip=True)
                            phd_info = f"{institution_name}, {department}, {extra_text}"
                        else:
                            phd_info = f"{institution_name}, {extra_text}"
                        
                        details['phd_institution'] = phd_info
        
    except requests.exceptions.RequestException as e:
        details['error'] = f"Request error: {str(e)}"
    except Exception as e:
        details['error'] = f"Parsing error: {str(e)}"
    
    return details


def scrape_all_profiles(profile_urls: List[str], output_file: str = OUTPUT_FILE, 
                       start_from: int = 0, save_interval: int = SAVE_INTERVAL):
    """
    Scrape details for all profiles and save incrementally.
    
    Args:
        profile_urls: List of profile URLs to scrape
        output_file: Path to output JSON file
        start_from: Index to start from (useful for resuming)
        save_interval: Save progress every N profiles
    """
    # Load existing data if available (for resuming)
    all_details = []
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            all_details = json.load(f)
        print(f"Loaded {len(all_details)} existing profiles from {output_file}")
        start_from = len(all_details)
    
    total = len(profile_urls)
    
    for i in range(start_from, total):
        url = profile_urls[i]
        print(f"\n[{i+1}/{total}] Scraping: {url}")
        
        details = scrape_profile_details(url)
        all_details.append(details)
        
        # Print summary
        print(f"  Specializations: {len(details['areas_of_specialization'])}")
        print(f"  Interests: {len(details['areas_of_interest'])}")
        print(f"  Current Institution: {details['current_institution']}")
        print(f"  PhD: {details['phd_institution']}")
        if details['error']:
            print(f"  ERROR: {details['error']}")
        
        # Save progress periodically
        if (i + 1) % save_interval == 0:
            with open(output_file, 'w') as f:
                json.dump(all_details, f, indent=2)
            print(f"\n✓ Progress saved ({i+1}/{total} completed)")
        
        # Rate limiting
        time.sleep(REQUEST_DELAY)
    
    # Final save
    with open(output_file, 'w') as f:
        json.dump(all_details, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ All profiles scraped! Results saved to {output_file}")
    print(f"Total profiles: {len(all_details)}")
    
    # Print statistics
    with_spec = sum(1 for d in all_details if d['areas_of_specialization'])
    with_interest = sum(1 for d in all_details if d['areas_of_interest'])
    with_phd = sum(1 for d in all_details if d['phd_institution'])
    with_errors = sum(1 for d in all_details if d['error'])
    
    print(f"\nStatistics:")
    print(f"  - With areas of specialization: {with_spec} ({with_spec/len(all_details)*100:.1f}%)")
    print(f"  - With areas of interest: {with_interest} ({with_interest/len(all_details)*100:.1f}%)")
    print(f"  - With PhD info: {with_phd} ({with_phd/len(all_details)*100:.1f}%)")
    print(f"  - With errors: {with_errors} ({with_errors/len(all_details)*100:.1f}%)")
    
    return all_details


def test_single_profile(url: str):
    """Test scraping on a single profile URL."""
    print(f"Testing profile scraping on: {url}")
    print("="*60)
    
    details = scrape_profile_details(url)
    
    print("\nResults:")
    print(json.dumps(details, indent=2))
    
    return details


if __name__ == "__main__":
    import sys
    
    # If a URL is provided as argument, test on that single profile
    if len(sys.argv) > 1:
        test_url = sys.argv[1]
        test_single_profile(test_url)
    else:
        # Load profile URLs
        if not os.path.exists(INPUT_FILE):
            print(f"Error: {INPUT_FILE} not found")
            print("Please provide a JSON file with a list of profile URLs to scrape")
            sys.exit(1)
            
        with open(INPUT_FILE, 'r') as f:
            profile_urls = json.load(f)
        
        print(f"Found {len(profile_urls)} profiles to scrape")
        print(f"Estimated time: {len(profile_urls) * REQUEST_DELAY / 3600:.1f} hours")
        print("\nStarting in 5 seconds... (Press Ctrl+C to cancel)")
        time.sleep(5)
        
        # Start scraping
        scrape_all_profiles(profile_urls, output_file=OUTPUT_FILE, save_interval=SAVE_INTERVAL)

