import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import urllib.parse

def get_abstract_from_openalex(title, doi_url=None):
    """
    Fetches the abstract using the OpenAlex API.
    Prioritizes DOI if available, otherwise searches by title.
    """
    base_url = "https://api.openalex.org/works"
    
    # Try to use DOI first (more accurate)
    if doi_url and "doi.org" in doi_url:
        doi = doi_url.split("doi.org/")[-1]
        url = f"{base_url}/https://doi.org/{doi}"
    else:
        # Fallback to search by title
        safe_title = urllib.parse.quote(title)
        url = f"{base_url}?search={safe_title}"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            
            # If searching by title, data is a list; take the first result
            if 'results' in data:
                if not data['results']:
                    return "Abstract not found"
                result = data['results'][0]
            else:
                result = data

            # OpenAlex stores abstracts as an inverted index to save space
            # We must reconstruct the text
            index = result.get('abstract_inverted_index')
            if index:
                abstract_words = {}
                for word, positions in index.items():
                    for pos in positions:
                        abstract_words[pos] = word
                # Join words in correct order
                full_abstract = " ".join([abstract_words[i] for i in sorted(abstract_words.keys())])
                return full_abstract
            return "No abstract available in OpenAlex"
    except Exception as e:
        return f"Error fetching abstract: {e}"
    
    return "Abstract not found"

def scrape_facct_papers(years):
    all_papers = []

    for year in years:
        url = f"https://facctconference.org/{year}/acceptedpapers.html"
        print(f"Scraping FAccT {year} from {url}...")
        
        try:
            response = requests.get(url)
            if response.status_code != 200:
                print(f"Failed to load page for {year}")
                continue
                
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # FAccT website structure varies slightly by year
            # 2022-2025 often use 'paper-entry' or specific list structures
            # 2020-2021 often use simple <ul> lists
            
            papers_found_in_year = 0
            
            # Strategy 1: Look for the modern 'paper-entry' divs (common in recent years)
            entries = soup.find_all('div', class_='paper-entry')
            
            # Strategy 2: If no divs, look for list items with links (older years)
            if not entries:
                # Target the main content area to avoid nav links
                content_div = soup.find('div', id='main-content') or soup.find('div', class_='container')
                if content_div:
                    # Find <li> elements that contain links
                    entries = content_div.find_all('li')

            for entry in entries:
                # Extract Title
                # Try finding a bold tag or a link first
                title_tag = entry.find('a') or entry.find('b')
                
                if title_tag:
                    title = title_tag.get_text(strip=True)
                    
                    # Basic filtering to skip navigation items/noise
                    if len(title) < 10 or "Session" in title or "Keynote" in title:
                        continue

                    # Extract DOI link if it exists (usually in the <a> href)
                    link = title_tag.get('href') if title_tag.name == 'a' else None
                    if not link and entry.find('a'):
                        link = entry.find('a').get('href')
                    
                    # Clean up link to ensure it's a DOI
                    doi_url = link if link and "doi.org" in link else None

                    # Extract Authors (simple text extraction based on typical formatting)
                    # Authors usually follow the title or are in the text of the <li>
                    full_text = entry.get_text(strip=True)
                    authors = full_text.replace(title, "").strip(" .(),")

                    # Fetch Abstract
                    # Adding a small delay to be polite to the API
                    time.sleep(0.2) 
                    abstract = get_abstract_from_openalex(title, doi_url)
                    
                    all_papers.append({
                        "Year": year,
                        "Title": title,
                        "Authors": authors,
                        "Abstract": abstract,
                        "DOI": doi_url
                    })
                    papers_found_in_year += 1

            print(f"  Found {papers_found_in_year} papers for {year}.")

        except Exception as e:
            print(f"  Error processing {year}: {e}")

    return pd.DataFrame(all_papers)

# --- EXECUTION ---
years_to_scrape = range(2020, 2026) # 2020 to 2025
df = scrape_facct_papers(years_to_scrape)

# Save to CSV
output_file = "facct_papers_2020_2025_with_abstracts.csv"
df.to_csv(output_file, index=False)
print(f"\nDone! Scraped {len(df)} papers. Data saved to {output_file}")
print(df.head())