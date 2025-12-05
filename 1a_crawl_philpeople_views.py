#!/usr/bin/env python
"""
Download HTML files from views URLs in survey_responses_filtered_reprocessed.json
Uses Selenium with rate limiting to avoid overwhelming the server.

Requirements:
    pip install selenium webdriver-manager beautifulsoup4

Usage:
    python 1_crawl_philpapers_views.py
    
    Or with a custom input file:
    python 1_crawl_philpapers_views.py --input custom_urls.json --output ./html_output
"""

import json
import os
import sys
import time
import random
import argparse
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Try to import webdriver_manager, fallback to system chromedriver if not available
try:
    from webdriver_manager.chrome import ChromeDriverManager
    USE_WEBDRIVER_MANAGER = True
except ImportError:
    USE_WEBDRIVER_MANAGER = False
    print("Note: webdriver_manager not installed, will use system chromedriver")
    print("Install with: pip install webdriver-manager")

from bs4 import BeautifulSoup

# Configuration (use relative paths from script directory)
INPUT_JSON = "./survey_responses_filtered_reprocessed.json"
OUTPUT_DIR = "./views_html"
MIN_DELAY = 2  # Minimum seconds between requests
MAX_DELAY = 5  # Maximum seconds between requests
TIMEOUT = 15  # Seconds to wait for page load

# For testing: set to a number to limit crawls, or None for all
MAX_URLS = None  # e.g., MAX_URLS = 5 for testing


def setup_driver():
    """Setup Selenium Chrome driver with options"""
    chrome_options = Options()
    chrome_options.add_argument('--headless')  # Run in background
    chrome_options.add_argument('--no-sandbox')
    chrome_options.add_argument('--disable-dev-shm-usage')
    chrome_options.add_argument('--disable-gpu')
    chrome_options.add_argument('--window-size=1920,1080')
    chrome_options.add_argument('--disable-blink-features=AutomationControlled')
    chrome_options.add_experimental_option('excludeSwitches', ['enable-automation'])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument('user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')

    try:
        if USE_WEBDRIVER_MANAGER:
            service = Service(ChromeDriverManager().install())
            driver = webdriver.Chrome(service=service, options=chrome_options)
        else:
            # Fallback to system chromedriver
            driver = webdriver.Chrome(options=chrome_options)
        
        # Additional settings to avoid detection
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        return driver
    except Exception as e:
        print(f"Error setting up Selenium driver: {e}")
        print("\nTroubleshooting:")
        print("  1. Ensure Chrome browser is installed")
        print("  2. Install dependencies: pip install selenium webdriver-manager")
        print("  3. If using system chromedriver, ensure it's in PATH")
        raise

def get_profile_slug(url):
    """Extract profile slug from URL"""
    # e.g., https://philpeople.org/profiles/sean-aas/views -> sean-aas
    return url.split('/')[-2]

def download_html(driver, url, output_path):
    """Download HTML from URL using Selenium"""
    try:
        driver.get(url)

        # Wait for the page to load (wait for body element)
        WebDriverWait(driver, TIMEOUT).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )

        # Additional small wait to ensure dynamic content loads
        time.sleep(1)

        # Get page source
        html = driver.page_source

        # Save to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)

        return True, len(html)

    except Exception as e:
        return False, str(e)

def main():
    """Main function to download all views HTML files"""

    # Load JSON data
    print(f"Loading data from {INPUT_JSON}...")
    with open(INPUT_JSON, 'r') as f:
        data = json.load(f)

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Extract URLs
    urls_to_download = []
    for entry in data:
        if 'views_url' in entry and entry['views_url']:
            urls_to_download.append(entry['views_url'])

    print(f"Found {len(urls_to_download)} URLs to download")

    # Check which files already exist
    existing_files = set(os.listdir(OUTPUT_DIR))
    urls_needed = []
    for url in urls_to_download:
        slug = get_profile_slug(url)
        filename = f"{slug}_views.html"
        if filename not in existing_files:
            urls_needed.append(url)

    if not urls_needed:
        print("All files already downloaded!")
        return

    # Apply limit if set
    if MAX_URLS is not None and MAX_URLS > 0:
        urls_needed = urls_needed[:MAX_URLS]
        print(f"Limiting to {MAX_URLS} URLs (for testing)")

    print(f"{len(urls_needed)} files need to be downloaded (skipping {len(urls_to_download) - len(urls_needed)} existing files)")

    # Setup Selenium driver
    print("Setting up Selenium driver...")
    driver = setup_driver()

    # Download with progress tracking
    success_count = 0
    fail_count = 0
    failed_urls = []

    try:
        for i, url in enumerate(urls_needed, 1):
            slug = get_profile_slug(url)
            filename = f"{slug}_views.html"
            output_path = os.path.join(OUTPUT_DIR, filename)

            print(f"[{i}/{len(urls_needed)}] Downloading {url}...")

            success, result = download_html(driver, url, output_path)

            if success:
                success_count += 1
                print(f"  ✓ Saved {result} bytes to {filename}")
            else:
                fail_count += 1
                failed_urls.append((url, result))
                print(f"  ✗ Failed: {result}")

            # Rate limiting (don't delay after the last request)
            if i < len(urls_needed):
                delay = random.uniform(MIN_DELAY, MAX_DELAY)
                print(f"  Waiting {delay:.1f} seconds...")
                time.sleep(delay)

    finally:
        driver.quit()

    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)
    print(f"Total URLs: {len(urls_needed)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")

    if failed_urls:
        print("\nFailed URLs:")
        for url, error in failed_urls:
            print(f"  - {url}")
            print(f"    Error: {error}")

    # Save failed URLs to a file for retry
    if failed_urls:
        failed_file = os.path.join(OUTPUT_DIR, "failed_downloads.json")
        with open(failed_file, 'w') as f:
            json.dump([{"url": url, "error": error} for url, error in failed_urls], f, indent=2)
        print(f"\nFailed URLs saved to {failed_file}")

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Download HTML files from PhilPeople profile views URLs'
    )
    parser.add_argument(
        '--input', '-i',
        default=INPUT_JSON,
        help=f'Input JSON file with views_url entries (default: {INPUT_JSON})'
    )
    parser.add_argument(
        '--output', '-o',
        default=OUTPUT_DIR,
        help=f'Output directory for HTML files (default: {OUTPUT_DIR})'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Limit number of URLs to crawl (for testing)'
    )
    parser.add_argument(
        '--delay-min',
        type=float,
        default=MIN_DELAY,
        help=f'Minimum delay between requests in seconds (default: {MIN_DELAY})'
    )
    parser.add_argument(
        '--delay-max',
        type=float,
        default=MAX_DELAY,
        help=f'Maximum delay between requests in seconds (default: {MAX_DELAY})'
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    # Override globals with command line args
    INPUT_JSON = args.input
    OUTPUT_DIR = args.output
    MIN_DELAY = args.delay_min
    MAX_DELAY = args.delay_max
    MAX_URLS = args.limit
    
    main()
