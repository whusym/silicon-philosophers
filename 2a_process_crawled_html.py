#!/usr/bin/env python3
"""
Extract survey responses from HTML files in views_html_filtered folder.
Output format: {philosopher: str, responses: {question: [answer]}}
"""

import os
import json
import re
from bs4 import BeautifulSoup
from pathlib import Path

def extract_philosopher_name(filename):
    """Extract philosopher name from filename like 'aaron-segal_views.html' -> 'aaron-segal'"""
    return filename.replace('_views.html', '')

def parse_response_cell(cell):
    """Parse a response cell and return list of answers"""
    answers = []
    
    # Get all text content first to check for combination answers
    cell_text = str(cell)
    
    # Check for "Accept a combination of answers:" followed by <ul>
    if 'Accept a combination of answers:' in cell_text:
        # Find all <li> elements
        list_items = cell.find_all('li')
        for li in list_items:
            # Extract text nodes and strong elements
            parts = []
            for content in li.contents:
                if isinstance(content, str):
                    text = content.strip()
                    if text:
                        parts.append(text)
                elif hasattr(content, 'name') and content.name == 'strong' and content.get('class') == ['thesis-title']:
                    parts.append(content.get_text(strip=True))
            
            if len(parts) >= 2:
                # Format: "prefix: thesis" (lowercase, with space)
                prefix = ' '.join(parts[0].split()).lower().strip()  # Normalize whitespace
                thesis = parts[1].strip()
                answer = f"{prefix}: {thesis}"
            elif len(parts) == 1:
                answer = ' '.join(parts[0].split()).lower().strip()
            else:
                continue
            
            answers.append(answer)
    else:
        # Single answer case
        # Look for <span> and <strong class="thesis-title">
        span = cell.find('span')
        strong = cell.find('strong', class_='thesis-title')
        
        if span and strong:
            span_text = span.get_text(strip=True)
            strong_text = strong.get_text(strip=True)
            # Remove trailing colon from span if present, then add colon before thesis
            # Single answers: keep original case, no space (e.g., "Accept:yes")
            span_text = span_text.rstrip(':').strip()
            strong_text = strong_text.strip()
            answer = f"{span_text}:{strong_text}"
            answers.append(answer)
        else:
            # Plain text response (like "Agnostic/undecided", "Skipped")
            # Get all text, but skip empty lines
            text = cell.get_text(separator=' ', strip=True)
            # Clean up multiple spaces
            text = ' '.join(text.split())
            if text:
                answers.append(text)
    
    return answers

def extract_survey_responses(html_file_path):
    """Extract survey responses from an HTML file"""
    with open(html_file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    soup = BeautifulSoup(content, 'html.parser')
    
    # Find the "My Survey Responses" heading
    h2 = soup.find('h2', string=re.compile('My Survey Responses'))
    if not h2:
        return None
    
    # Find the table after the h2
    table = h2.find_next('table')
    if not table:
        return None
    
    # Extract philosopher name from filename
    filename = os.path.basename(html_file_path)
    philosopher = extract_philosopher_name(filename)
    
    responses = {}
    
    # Find all rows in tbody
    tbody = table.find('tbody')
    if not tbody:
        return None
    
    rows = tbody.find_all('tr')
    
    for row in rows:
        cells = row.find_all('td')
        if len(cells) < 2:
            continue
        
        # First cell contains the question
        question_cell = cells[0]
        question_link = question_cell.find('a', class_='survey-question-title')
        if not question_link:
            continue
        
        question = question_link.get_text(strip=True).lower()
        
        # Second cell contains the response
        response_cell = cells[1]
        answers = parse_response_cell(response_cell)
        
        if answers:
            responses[question] = answers
    
    return {
        'philosopher': philosopher,
        'responses': responses
    }

def main():
    """Process all HTML files in views_html_filtered folder"""
    input_dir = Path('views_html_filtered')
    
    if not input_dir.exists():
        print(f"Error: Directory {input_dir} does not exist")
        return
    
    results = []
    
    # Process each HTML file
    html_files = sorted(input_dir.glob('*_views.html'))
    print(f"Found {len(html_files)} HTML files to process")
    
    for html_file in html_files:
        print(f"Processing {html_file.name}...")
        try:
            result = extract_survey_responses(html_file)
            if result and result['responses']:
                results.append(result)
            else:
                print(f"  Warning: No survey responses found in {html_file.name}")
        except Exception as e:
            print(f"  Error processing {html_file.name}: {e}")
    
    # Write results to JSON file
    output_file = 'philosopher_survey_responses.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\nExtraction complete! Results written to {output_file}")
    print(f"Total philosophers processed: {len(results)}")

if __name__ == '__main__':
    main()

