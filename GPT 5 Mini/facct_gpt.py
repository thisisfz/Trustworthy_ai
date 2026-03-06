import pandas as pd
import openai
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# 1. Load the scraped papers
df = pd.read_csv("facct_papers_final.csv")

# 2. Configure Lab Proxy
client = openai.OpenAI(api_key="sk-I56Y3797SKvnhrmXYAVTNQ", base_url="http://131.220.150.238:8080/")

# 3. Define Taxonomy based on Grace et al.
taxonomy = ["Authoritarianism", "Bias & Inequality", "Disempowerment", "Misinformation", "Robustness", "Extinction Risk"]

def classify_paper(title, abstract):
    # Handle missing abstracts just in case
    abstract_text = str(abstract) if pd.notna(abstract) else "No abstract available"
    
    # Updated prompt to include Abstract
    prompt = (
        f"Classify this paper into one category: {', '.join(taxonomy)}.\n\n"
        f"Title: {title}\n"
        f"Abstract: {abstract_text}\n\n"
        f"Category:"
    )
    
    try:
        response = client.chat.completions.create(
            model="openai/gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"

# Helper wrapper for parallel execution
def process_row(row):
    return classify_paper(row['Title'], row['Abstract'])

# 4. Run Classification (Parallelized)
print(f"Classifying {len(df)} papers...")

# Convert rows to a list for iteration
rows = [row for _, row in df.iterrows()]

# Execute in parallel with 10 workers
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(tqdm(executor.map(process_row, rows), total=len(rows)))

df['Grace_Category'] = results
df.to_csv("facct_2025_classified_heute1.csv", index=False)