# extract_zip.py

import zipfile
import os

zip_file = "dogs-vs-cats.zip"  # Ya jo bhi naam hai

print(f"🔄 Extracting {zip_file}...")

with zipfile.ZipFile(zip_file, 'r') as zip_ref:
    zip_ref.extractall("data/")

print("✅ Extracted to data/ folder")

# Check karo
import os
for root, dirs, files in os.walk("data/", topdown=True):
    level = root.replace("data/", "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 2 * (level + 1)
    for file in files[:3]:  # Sirf pehle 3 files
        print(f"{subindent}{file}")
    if len(files) > 3:
        print(f"{subindent}... and {len(files)-3} more files")