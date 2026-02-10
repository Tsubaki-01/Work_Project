import json

import pandas as pd

from silver_pilot.config import config

df = pd.read_csv(config.DATA_DIR / "raw" / "datasets" / "full_export.csv")

unique_labels = df["_labels"].dropna().unique().tolist()
unique_types = df["_type"].dropna().unique().tolist()

data_to_save = {"labels": unique_labels, "types": unique_types}

output_filename = config.DATA_DIR / "processed" / "extract" / "neo4j_schema.json"
with open(output_filename, "w", encoding="utf-8") as f:
    json.dump(data_to_save, f, ensure_ascii=False, indent=4)

print(f"提取完成！文件已保存为: {output_filename}")
