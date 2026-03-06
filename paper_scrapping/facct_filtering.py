import requests
import pandas as pd
import time
import urllib.parse

def get_openalex_abstract(title, doi):
    """
    Fetches abstract from OpenAlex using DOI (best) or Title (fallback).
    """
    base_url = "https://api.openalex.org/works"
    
    # 1. Best Method: Search by DOI
    if doi:
        # Clean DOI to ensure it's just the ID (e.g. 10.1145/XXXXX)
        clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "")
        url = f"{base_url}/https://doi.org/{clean_doi}"
    else:
        # 2. Fallback: Search by Title
        url = f"{base_url}?search={urllib.parse.quote(title)}"

    try:
        r = requests.get(url)
        if r.status_code == 200:
            data = r.json()
            
            # Handle search results vs direct object retrieval
            if 'results' in data:
                if not data['results']: return None
                item = data['results'][0]
            else:
                item = data

            # Reconstruct abstract from inverted index
            index = item.get('abstract_inverted_index')
            if index:
                words = {}
                for word, positions in index.items():
                    for pos in positions:
                        words[pos] = word
                return " ".join([words[i] for i in sorted(words.keys())])
    except:
        pass
    return None

def get_dblp_papers(year):
    """
    Dynamically finds the correct DBLP conference key and fetches papers.
    """
    print(f"Processing Year: {year}...")
    
    # 1. Search for the conference venue to get the correct key
    # FAccT was "FAT*" in 2020, "FAccT" from 2021 onwards
    q = f"FAT* {year}" if year == 2020 else f"FAccT {year}"
    search_url = f"https://dblp.org/search/publ/api?q={q}&format=json&h=1000"
    
    try:
        r = requests.get(search_url)
        data = r.json()
        hits = data.get('result', {}).get('hits', {}).get('hit', [])
    except Exception as e:
        print(f"  Error searching DBLP: {e}")
        return []

    papers = []
    print(f"  Found {len(hits)} potential entries in DBLP. Filtering...")

    for hit in hits:
        info = hit.get('info', {})
        
        # 2. Filter for actual conference papers
        # We skip "Front Matter", "Editors", "Keynotes" by checking type or title
        if info.get('type') != 'Conference and Workshop Papers':
            continue
            
        title = info.get('title', '')
        if any(x in title for x in ['Front Matter', 'Message from', 'Keynote', 'Session:']):
            continue

        # Extract Authors
        authors_data = info.get('authors', {}).get('author', [])
        if isinstance(authors_data, list):
            authors = ", ".join([a.get('text', '') for a in authors_data])
        elif isinstance(authors_data, dict):
            authors = authors_data.get('text', '')
        else:
            authors = ""

        papers.append({
            "Year": year,
            "Title": title,
            "Authors": authors,
            "DOI": info.get('doi', ''),
            "URL": info.get('url', '')
        })

    return papers

# --- MAIN SCRIPT ---
all_data = []
for year in range(2020, 2026):
    year_papers = get_dblp_papers(year)
    
    # Fetch abstracts for found papers
    print(f"  Fetching abstracts for {len(year_papers)} papers...")
    for i, p in enumerate(year_papers):
        # simple progress counter
        if i % 10 == 0: print(f"    {i}/{len(year_papers)}", end='\r')
        
        p['Abstract'] = get_openalex_abstract(p['Title'], p['DOI'])
        time.sleep(0.1) # Be polite to API
        
    all_data.extend(year_papers)
    print(f"  Done with {year}.\n")

# Save
df = pd.DataFrame(all_data)
df.to_csv("facct_papers_final.csv", index=False)
print(f"Completed! Saved {len(df)} papers to facct_papers_final.csv")