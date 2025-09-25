import zipfile
import os

folder = "gpt2-finetuned"
zip_filename = "gpt2-finetuned.zip"

with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk(folder):
        for file in files:
            filepath = os.path.join(root, file)
            # Ensure forward slashes in the archive
            arcname = os.path.relpath(filepath, folder).replace("\\", "/")
            zipf.write(filepath, os.path.join(folder, arcname))

print("✅ Zipped successfully: gpt2-finetuned.zip")
