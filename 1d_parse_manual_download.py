#!/usr/bin/env python3
"""
Parse manually downloaded PhilPapers respondents page and crawl profiles
Works around Cloudflare protection by using manually downloaded HTML
"""

import requests
from bs4 import BeautifulSoup
import json
import time
import os
from urllib.parse import urljoin
import re
import sys

class ManualPhilPapersCrawler:
    def __init__(self, delay=3):
        self.delay = delay
        self.base_url = "https://philpapers.org/surveys/"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        })

    def parse_local_html(self, html_file):
        """Parse locally saved HTML file to extract myview.html links"""
        print(f"Parsing local file: {html_file}")

        if not os.path.exists(html_file):
            print(f"Error: File not found: {html_file}")
            print("\nPlease:")
            print("1. Visit https://philpapers.org/surveys/public_respondents.html in your browser")
            print("2. Save the page (Cmd+S) as 'respondents_manual.html'")
            print("3. Place it in the philpapers_survey folder")
            return []

        with open(html_file, 'r', encoding='utf-8') as f:
            html = f.read()

        soup = BeautifulSoup(html, 'html.parser')

        # Check if this is a Cloudflare block page
        if 'cloudflare' in html.lower() and 'blocked' in html.lower():
            print("⚠️  This appears to be a Cloudflare block page")
            print("Please save the actual content page after it loads in your browser")
            return []

        # Find all links
        myview_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']

            # Look for myview.html links
            if 'myview.html' in href:
                # Construct full URL
                if href.startswith('http'):
                    full_url = href
                elif href.startswith('/'):
                    full_url = 'https://philpapers.org' + href
                else:
                    full_url = urljoin(self.base_url, href)

                link_text = link.get_text(strip=True)

                myview_links.append({
                    'url': full_url,
                    'name': link_text,
                    'href': href
                })

        print(f"Found {len(myview_links)} myview.html links")
        return myview_links

    def fetch_page(self, url):
        """Fetch a page with rate limiting"""
        try:
            print(f"  Fetching: {url}")
            response = self.session.get(url, timeout=30)

            # Check if we got blocked
            if response.status_code == 403:
                print(f"  ⚠️  Got 403 Forbidden - site may be blocking us")
                return None

            response.raise_for_status()
            time.sleep(self.delay)
            return response.text

        except requests.RequestException as e:
            print(f"  Error: {e}")
            return None

    def parse_myview_page(self, url, name):
        """Parse a myview.html page"""
        html = self.fetch_page(url)
        if not html:
            return None

        soup = BeautifulSoup(html, 'html.parser')

        # Extract ID
        philosopher_id = url.split('/')[-2] if '/' in url else 'unknown'

        # Extract title
        title = None
        h1 = soup.find('h1')
        if h1:
            title = h1.get_text(strip=True)

        # Extract content
        content_text = soup.get_text(separator='\n', strip=True)

        # Extract survey responses
        responses = []

        # Try tables
        for table in soup.find_all('table'):
            for row in table.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                if len(cells) >= 2:
                    question = cells[0].get_text(strip=True)
                    answer = cells[1].get_text(strip=True)
                    if question and answer and len(question) > 3:
                        responses.append({
                            'question': question,
                            'answer': answer
                        })

        # Try definition lists
        for dl in soup.find_all('dl'):
            dts = dl.find_all('dt')
            dds = dl.find_all('dd')
            for dt, dd in zip(dts, dds):
                question = dt.get_text(strip=True)
                answer = dd.get_text(strip=True)
                if question and answer:
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
            'full_html': html,
            'crawled_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }

    def crawl_from_manual_html(self, html_file, output_dir):
        """Main crawl function using manually downloaded HTML"""
        print("=" * 80)
        print("PhilPapers Survey Crawler (Manual HTML Mode)")
        print("=" * 80)

        os.makedirs(output_dir, exist_ok=True)

        # Parse local HTML
        myview_links = self.parse_local_html(html_file)

        if not myview_links:
            print("\n❌ No links found")
            return

        # Save links
        links_file = os.path.join(output_dir, '_myview_links.json')
        with open(links_file, 'w', encoding='utf-8') as f:
            json.dump(myview_links, f, indent=2, ensure_ascii=False)
        print(f"Saved links list to: {links_file}")

        # Crawl profiles
        print(f"\nCrawling {len(myview_links)} profiles with {self.delay}s delay...")
        print("-" * 80)

        successful = 0
        failed = 0

        for i, link_info in enumerate(myview_links, 1):
            url = link_info['url']
            name = link_info['name']

            print(f"\n[{i}/{len(myview_links)}] {name}")

            try:
                data = self.parse_myview_page(url, name)

                if not data:
                    failed += 1
                    continue

                # Create filename
                filename = data['id']
                if filename == 'unknown':
                    filename = re.sub(r'[^\w\s-]', '', name).strip().replace(' ', '_')[:50]

                # Save JSON
                json_file = os.path.join(output_dir, f"{filename}.json")
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

                # Save HTML
                html_file = os.path.join(output_dir, f"{filename}.html")
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(data['full_html'])

                print(f"  ✓ Saved: {filename}")
                if data['responses']:
                    print(f"  Survey responses: {len(data['responses'])}")

                successful += 1

            except Exception as e:
                print(f"  ✗ Error: {e}")
                failed += 1

        # Summary
        print("\n" + "=" * 80)
        print("Crawl Complete!")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Total: {len(myview_links)}")
        print(f"Output directory: {output_dir}")
        print("=" * 80)


def main():
    # Look for manually downloaded HTML file
    possible_files = [
        'respondents_manual.html',
        'public_respondents.html',
        'respondents.html'
    ]

    html_file = None
    for filename in possible_files:
        if os.path.exists(filename):
            html_file = filename
            break

    if not html_file:
        print("=" * 80)
        print("Manual HTML File Not Found")
        print("=" * 80)
        print("\nTo use this crawler:")
        print("\n1. Open your browser and visit:")
        print("   https://philpapers.org/surveys/public_respondents.html")
        print("\n2. Wait for the page to fully load")
        print("\n3. Save the page:")
        print("   - Right-click → 'Save As...'")
        print("   - Or press Cmd+S (Mac) / Ctrl+S (Windows)")
        print("   - Save as: respondents_manual.html")
        print("\n4. Place the file in the philpapers_survey folder")
        print("\n5. Run this script again")
        print("=" * 80)
        sys.exit(1)

    output_dir = "./philosopher_profiles"
    delay = 3

    crawler = ManualPhilPapersCrawler(delay=delay)
    crawler.crawl_from_manual_html(html_file, output_dir)


if __name__ == "__main__":
    main()
