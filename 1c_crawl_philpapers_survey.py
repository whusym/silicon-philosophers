#!/usr/bin/env python3
"""
PhilPapers Survey Crawler
Crawls philosopher responses from philpapers.org/surveys/public_respondents.html

Uses Selenium for crawling since requests are blocked (403 Forbidden).

Cloudflare Handling:
    If blocked by Cloudflare in headless mode, the crawler will automatically:
    1. Restart in visible browser mode
    2. Wait for you to solve any CAPTCHA challenges
    3. Continue crawling after the challenge is passed

Requirements:
    pip install selenium webdriver-manager beautifulsoup4

Usage:
    python 1c_crawl_philpapers_survey.py
    
    # With options:
    python 1c_crawl_philpapers_survey.py --limit 10 --output ./my_profiles
    
    # Start with visible browser (skip headless):
    python 1c_crawl_philpapers_survey.py --visible
    
    # Disable automatic Cloudflare retry:
    python 1c_crawl_philpapers_survey.py --no-cloudflare-retry
    
Troubleshooting Cloudflare blocks:
    1. Use --visible to start with visible browser
    2. Try increasing --delay to 5-10 seconds
    3. Use a VPN or different IP
    4. Manually download pages and use 1d_parse_manual_download.py
"""

import json
import time
import os
import re
import argparse
from urllib.parse import urljoin

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


class PhilPapersCrawler:
    """Crawler for PhilPapers survey respondents using Selenium"""
    
    def __init__(self, base_url="https://philpapers.org/surveys/", delay=3, timeout=15, headless=True):
        self.base_url = base_url
        self.delay = delay
        self.timeout = timeout
        self.driver = None
        self.headless = headless
        self.cloudflare_retry_with_visible = True  # Retry with visible browser on Cloudflare block
    
    def setup_driver(self, headless=None):
        """Setup Selenium Chrome driver with options"""
        if headless is None:
            headless = self.headless
            
        chrome_options = Options()
        if headless:
            chrome_options.add_argument('--headless')
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
                self.driver = webdriver.Chrome(service=service, options=chrome_options)
            else:
                self.driver = webdriver.Chrome(options=chrome_options)
            
            # Additional settings to avoid detection
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            mode = "headless" if headless else "visible"
            print(f"✓ Selenium driver initialized ({mode} mode)")
        except Exception as e:
            print(f"Error setting up Selenium driver: {e}")
            print("\nTroubleshooting:")
            print("  1. Ensure Chrome browser is installed")
            print("  2. Install dependencies: pip install selenium webdriver-manager")
            raise

    def close_driver(self):
        """Close the Selenium driver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
            print("✓ Selenium driver closed")
    
    def restart_driver_visible(self):
        """Restart driver in visible (non-headless) mode to bypass Cloudflare"""
        print("\n🔄 Restarting browser in visible mode to bypass Cloudflare...")
        self.close_driver()
        self.headless = False
        self.setup_driver(headless=False)
        print("✓ Browser restarted in visible mode")
        print("  If you see a Cloudflare challenge, please solve it manually.")
        print("  The crawler will continue automatically after the page loads.\n")

    def fetch_page(self, url, retry_on_cloudflare=True):
        """Fetch a page with error handling and rate limiting"""
        try:
            print(f"  Fetching: {url}")
            self.driver.get(url)
            WebDriverWait(self.driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            time.sleep(1)  # Allow dynamic content to load
            
            html = self.driver.page_source
            
            # Check for Cloudflare block and retry with visible browser
            if retry_on_cloudflare and self.is_cloudflare_blocked(html):
                if self.cloudflare_retry_with_visible and self.headless:
                    print("  ⚠ Cloudflare detected! Switching to visible browser...")
                    self.restart_driver_visible()
                    return self.fetch_page_with_cloudflare_wait(url)
            
            time.sleep(self.delay)  # Rate limiting
            return html
        except Exception as e:
            print(f"  Error fetching {url}: {e}")
            return None
    
    def fetch_page_with_cloudflare_wait(self, url, max_wait=60):
        """Fetch page with visible browser and wait for Cloudflare challenge to be solved"""
        try:
            print(f"  Fetching with visible browser: {url}")
            self.driver.get(url)
            
            # Wait for initial page load
            WebDriverWait(self.driver, self.timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Check if we're on a Cloudflare challenge page
            start_time = time.time()
            while time.time() - start_time < max_wait:
                html = self.driver.page_source
                
                if not self.is_cloudflare_blocked(html):
                    print("  ✓ Cloudflare challenge passed!")
                    time.sleep(self.delay)
                    return html
                
                # Still blocked, wait and check again
                elapsed = int(time.time() - start_time)
                print(f"\r  ⏳ Waiting for Cloudflare challenge... ({elapsed}s / {max_wait}s)", end='', flush=True)
                time.sleep(2)
            
            print(f"\n  ✗ Timeout waiting for Cloudflare challenge (>{max_wait}s)")
            return self.driver.page_source
            
        except Exception as e:
            print(f"  Error fetching {url}: {e}")
            return None

    def extract_myview_links(self, respondents_url, max_links=None):
        """Extract all links ending with myview.html from the respondents page"""
        print("\nFetching respondents page...")
        html = self.fetch_page(respondents_url)
        if not html:
            return []

        soup = BeautifulSoup(html, 'html.parser')
        myview_links = []

        # Find all links ending with myview.html
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.endswith('myview.html'):
                full_url = urljoin(self.base_url, href)
                link_text = link.get_text(strip=True)
                
                # Skip generic links
                if link_text.lower() in ['here', 'click here', 'view']:
                    continue

                myview_links.append({
                    'url': full_url,
                    'name': link_text,
                    'href': href
                })
                
                # Apply limit if set
                if max_links and len(myview_links) >= max_links:
                    break

        return myview_links

    def is_cloudflare_blocked(self, html):
        """Check if page is blocked by Cloudflare"""
        if not html:
            return False
        blocked_indicators = [
            'you have been blocked',
            'cloudflare ray id',
            'please enable cookies',
            'security service to protect itself',
        ]
        html_lower = html.lower()
        return any(indicator in html_lower for indicator in blocked_indicators)

    def parse_myview_page(self, url, name):
        """Parse a myview.html page and extract structured data"""
        html = self.fetch_page(url)
        if not html:
            return None

        # Check for Cloudflare block
        if self.is_cloudflare_blocked(html):
            print(f"  ⚠ Cloudflare blocked access to this page")
            return {'blocked': True, 'url': url, 'name': name}

        soup = BeautifulSoup(html, 'html.parser')

        # Extract philosopher ID from URL
        philosopher_id = url.split('/')[-2] if '/' in url else 'unknown'

        # Extract title/header
        title = None
        h1 = soup.find('h1')
        if h1:
            title = h1.get_text(strip=True)

        # Extract main content
        main_content = soup.find('div', id='main')
        content_text = ""
        if main_content:
            content_text = main_content.get_text(separator='\n', strip=True)
        else:
            # Try to get body content if no main div
            body = soup.find('body')
            if body:
                content_text = body.get_text(separator='\n', strip=True)

        # Extract all text content (full HTML)
        full_html = html

        # Look for survey responses (often in tables or lists)
        responses = []

        # Try to find tables
        tables = soup.find_all('table')
        for table in tables:
            rows = table.find_all('tr')
            for row in rows:
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    question = cells[0].get_text(strip=True)
                    answer = cells[1].get_text(strip=True)
                    if question and answer and len(question) > 3:
                        responses.append({
                            'question': question,
                            'answer': answer
                        })

        # Try to find definition lists (dl/dt/dd)
        dls = soup.find_all('dl')
        for dl in dls:
            dts = dl.find_all('dt')
            dds = dl.find_all('dd')
            for dt, dd in zip(dts, dds):
                question = dt.get_text(strip=True)
                answer = dd.get_text(strip=True)
                if question and answer and len(question) > 3:
                    responses.append({
                        'question': question,
                        'answer': answer
                    })

        return {
            'id': philosopher_id,
            'name': name,
            'url': url,
            'title': title,
            'content_text': content_text,
            'responses': responses,
            'full_html': full_html,
            'crawled_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }

    def crawl_all(self, respondents_url, output_dir, max_profiles=None):
        """Crawl all myview.html pages"""
        print("=" * 80)
        print("PhilPapers Survey Crawler (Selenium)")
        print("=" * 80)
        print(f"Mode: {'headless' if self.headless else 'visible browser'}")
        print(f"Cloudflare retry: {'enabled' if self.cloudflare_retry_with_visible else 'disabled'}")

        # Setup driver
        self.setup_driver()

        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)

            # Extract all myview links
            print("\nExtracting myview.html links...")
            myview_links = self.extract_myview_links(respondents_url, max_links=max_profiles)
            
            # If no links found, might be Cloudflare - retry with visible browser
            if not myview_links and self.headless and self.cloudflare_retry_with_visible:
                print("  ⚠ No links found - may be Cloudflare blocked")
                print("  Retrying with visible browser...")
                self.restart_driver_visible()
                myview_links = self.extract_myview_links(respondents_url, max_links=max_profiles)
            
            print(f"Found {len(myview_links)} philosopher profiles")

            if not myview_links:
                print("\n⚠ No profile links found!")
                print("This might be due to:")
                print("  - Page structure changed")
                print("  - Cloudflare blocking")
                print("  - Network issues")
                return

            # Save links list
            links_file = os.path.join(output_dir, '_myview_links.json')
            with open(links_file, 'w', encoding='utf-8') as f:
                json.dump(myview_links, f, indent=2, ensure_ascii=False)
            print(f"Saved links to: {links_file}")

            # Crawl each profile
            print(f"\nStarting crawl with {self.delay}s delay between requests...")
            print("-" * 80)

            successful = 0
            failed = 0
            blocked = 0

            for i, link_info in enumerate(myview_links, 1):
                url = link_info['url']
                name = link_info['name']

                print(f"\n[{i}/{len(myview_links)}] Crawling: {name}")

                # Parse the page
                try:
                    data = self.parse_myview_page(url, name)

                    if not data:
                        failed += 1
                        continue

                    # Check if blocked by Cloudflare
                    if data.get('blocked'):
                        blocked += 1
                        continue

                    # Create filename from ID or name
                    filename = data['id']
                    if filename == 'unknown':
                        # Use sanitized name as filename
                        filename = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')

                    # Save data (without full_html for smaller files)
                    data_to_save = {k: v for k, v in data.items() if k != 'full_html'}
                    json_file = os.path.join(output_dir, f"{filename}.json")
                    with open(json_file, 'w', encoding='utf-8') as f:
                        json.dump(data_to_save, f, indent=2, ensure_ascii=False)

                    # Also save raw HTML
                    html_file = os.path.join(output_dir, f"{filename}.html")
                    with open(html_file, 'w', encoding='utf-8') as f:
                        f.write(data['full_html'])

                    print(f"  ✓ Saved: {json_file}")
                    print(f"    Name: {name}")
                    if data['responses']:
                        print(f"    Responses found: {len(data['responses'])}")

                    successful += 1

                except Exception as e:
                    print(f"  ✗ Error processing {name}: {e}")
                    failed += 1

            # Summary
            print("\n" + "=" * 80)
            print("Crawl Complete!")
            print(f"Successful: {successful}")
            print(f"Failed: {failed}")
            if blocked > 0:
                print(f"Blocked by Cloudflare: {blocked}")
                print("\n⚠ Some pages were blocked by Cloudflare protection.")
                print("  Try: 1) Increasing --delay  2) Using a VPN  3) Manual download with 6c_parse_manual_download.py")
            print(f"Total: {len(myview_links)}")
            print(f"Output directory: {output_dir}")
            print("=" * 80)

        finally:
            self.close_driver()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Crawl PhilPapers survey respondent profiles'
    )
    parser.add_argument(
        '--output', '-o',
        default='./philosopher_profiles',
        help='Output directory for crawled data (default: ./philosopher_profiles)'
    )
    parser.add_argument(
        '--limit', '-l',
        type=int,
        default=None,
        help='Limit number of profiles to crawl (for testing)'
    )
    parser.add_argument(
        '--delay', '-d',
        type=int,
        default=3,
        help='Delay between requests in seconds (default: 3)'
    )
    parser.add_argument(
        '--url',
        default='https://philpapers.org/surveys/public_respondents.html',
        help='URL of the respondents page'
    )
    parser.add_argument(
        '--visible', '-v',
        action='store_true',
        help='Start with visible browser window (non-headless mode)'
    )
    parser.add_argument(
        '--no-cloudflare-retry',
        action='store_true',
        help='Disable automatic retry with visible browser on Cloudflare block'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Configuration
    respondents_url = args.url
    output_dir = args.output
    delay = args.delay
    max_profiles = args.limit
    headless = not args.visible
    cloudflare_retry = not args.no_cloudflare_retry

    if max_profiles:
        print(f"Limiting to {max_profiles} profiles (for testing)")
    
    if args.visible:
        print("Starting in visible browser mode")

    # Create crawler and run
    crawler = PhilPapersCrawler(delay=delay, headless=headless)
    crawler.cloudflare_retry_with_visible = cloudflare_retry
    crawler.crawl_all(respondents_url, output_dir, max_profiles=max_profiles)


if __name__ == "__main__":
    main()
