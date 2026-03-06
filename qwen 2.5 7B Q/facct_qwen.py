import pandas as pd
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

# 1. Load the papers
df = pd.read_csv("facct_papers_final.csv")

# 2. Taxonomy
taxonomy = ["Authoritarianism", "Bias & Inequality", "Disempowerment", 
            "Misinformation", "Robustness", "Extinction Risk"]

# 3. Optimize Ollama for your MacBook Air
os.environ['OLLAMA_NUM_GPU'] = '1'           # Enable GPU acceleration
os.environ['OLLAMA_GPU_LAYERS'] = '35'        # Optimize for 7B models
os.environ['OLLAMA_NUM_THREADS'] = '4'        # Match M2 performance cores

MODEL_NAME = "qwen2.5:7b-instruct"
OLLAMA_URL = "http://localhost:11434/api/generate"

def query_ollama(prompt):
    """Query Ollama with optimized settings"""
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 20,
            "num_ctx": 2048,           # Reduced context window saves memory
            "num_batch": 512,           # Optimal batch size for M2
            "num_gpu": 1                 # Force GPU usage
        }
    }
    
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
        if response.status_code == 200:
            return response.json().get('response', '').strip()
        return f"Error: HTTP {response.status_code}"
    except Exception as e:
        return f"Error: {str(e)}"

def classify_paper(title, abstract):
    """Classify a single paper"""
    abstract_text = str(abstract) if pd.notna(abstract) else "No abstract available"
    if len(abstract_text) > 1500:
        abstract_text = abstract_text[:1500] + "..."
    
    prompt = f"""Classify this paper into exactly one category:
{', '.join(taxonomy)}

Title: {title}
Abstract: {abstract_text}

Category:"""
    
    result = query_ollama(prompt)
    
    # Clean and match
    if "Error" not in result:
        result_lower = result.lower()
        for cat in taxonomy:
            if cat.lower() in result_lower:
                return cat
        return f"Unknown: {result[:30]}"
    return result

# MAIN EXECUTION
print("=" * 60)
print("OPTIMIZED CLASSIFICATION WITH GPU + MULTITHREADING")
print("=" * 60)

# Verify GPU is active
print("\n🔍 Checking GPU status...")
try:
    test_prompt = "Say OK"
    start = time.time()
    result = query_ollama(test_prompt)
    elapsed = time.time() - start
    
    if "Error" not in result:
        print(f"✅ GPU is active! Response time: {elapsed:.2f}s")
        print("   (Under 2 seconds indicates GPU acceleration)")
    else:
        print(f"❌ GPU check failed: {result}")
        exit(1)
except Exception as e:
    print(f"❌ Error: {e}")
    exit(1)

# Process with concurrency
print(f"\n📊 Processing {len(df)} papers with 3 concurrent workers...")
print("⏳ Estimated time: ~15-20 minutes total (vs 13-18 hours sequential)")

# Use ThreadPoolExecutor with 3 workers (optimal for 8GB RAM)
results = [None] * len(df)
successful = 0
errors = 0

with ThreadPoolExecutor(max_workers=3) as executor:
    # Submit all tasks
    future_to_idx = {
        executor.submit(classify_paper, row['Title'], row['Abstract']): idx 
        for idx, row in df.iterrows()
    }
    
    # Process as they complete
    for future in tqdm(as_completed(future_to_idx), total=len(df), desc="Classifying"):
        idx = future_to_idx[future]
        try:
            category = future.result(timeout=120)
            results[idx] = category
            
            if "Error" in category:
                errors += 1
            elif "Unknown" not in category:
                successful += 1
            
            # Save checkpoint every 25 papers
            if (idx + 1) % 25 == 0:
                df_temp = df.iloc[:idx+1].copy()
                df_temp['Grace_Category'] = results[:idx+1]
                df_temp.to_csv(f"checkpoint_{idx+1}.csv", index=False)
                
        except Exception as e:
            results[idx] = f"Error: Timeout"
            errors += 1

df['Grace_Category'] = results

# Save final results
output_file = "facct_2025_classified_optimized.csv"
df.to_csv(output_file, index=False)

# Summary
print("\n" + "=" * 60)
print("FINAL CLASSIFICATION SUMMARY")
print("=" * 60)
print(df['Grace_Category'].value_counts())

success = len(df) - errors - (df['Grace_Category'].str.contains('Unknown', na=False).sum())
print(f"\n📊 Statistics:")
print(f"  Total papers: {len(df)}")
print(f"  Successfully classified: {success} ({success/len(df)*100:.1f}%)")
print(f"  Errors: {errors} ({errors/len(df)*100:.1f}%)")
print(f"  Time per paper: ~{((time.time() - start)/len(df)):.2f}s average")
print(f"\n✅ Results saved to: {output_file}")