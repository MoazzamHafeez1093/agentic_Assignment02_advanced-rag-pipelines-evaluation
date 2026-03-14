import os
import urllib.request
import bz2
import shutil
import sys

def main():
    url = "https://github.com/facebookresearch/CRAG/raw/refs/heads/main/data/crag_task_1_and_2_dev_v4.jsonl.bz2"
    bz2_path = "dataset/crag.jsonl.bz2"
    out_path = "dataset/crag_task_1_and_2_dev_v4.jsonl"
    
    os.makedirs("dataset", exist_ok=True)
    
    print("Downloading dataset (this may take a few minutes)...")
    try:
        urllib.request.urlretrieve(url, bz2_path)
        print("Download complete. Extracting...")
        with bz2.open(bz2_path, 'rb') as f_in, open(out_path, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        print("Extraction complete. Dataset ready.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
