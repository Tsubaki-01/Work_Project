from tqdm import tqdm

from silver_pilot.config import config
from silver_pilot.tools.document import MarkdownCleaner

output_path = config.DATA_DIR / "processed/extract/milvus/md"

dir_path = config.DATA_DIR / "raw/databases/milvus"
file_paths = list(dir_path.rglob("*.md"))

cleaner = MarkdownCleaner()
for file_path in tqdm(file_paths, desc="Cleaning Markdown files"):
    cleaned_text = cleaner.clean(file_path, output_path / file_path.name)
