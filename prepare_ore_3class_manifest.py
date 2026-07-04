from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = ROOT / "remaining_dataset_clean" / "images_by_class"
OUT_ROOT = ROOT / "ore_3class"
MAPPING = {
    "ordinary": ("ordinary", "Рядовая руда"),
    "talc": ("talc", "Оталькованная руда"),
    "thin": ("difficult", "Труднообогатимая руда"),
    "refractory": ("difficult", "Труднообогатимая руда"),
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def main() -> None:
    OUT_ROOT.mkdir(exist_ok=True)
    rows = []
    for source_folder, (label, label_ru) in MAPPING.items():
        for path in sorted((SOURCE_ROOT / source_folder).glob("*")):
            if path.suffix.lower() in IMAGE_EXTS:
                rows.append({
                    "path": str(path.relative_to(ROOT)),
                    "source_folder": source_folder,
                    "label_3class": label,
                    "label_ru": label_ru,
                })
    with (OUT_ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "source_folder", "label_3class", "label_ru"])
        writer.writeheader()
        writer.writerows(rows)
    counts = {label: 0 for label in ["ordinary", "talc", "difficult"]}
    for row in rows:
        counts[row["label_3class"]] += 1
    with (OUT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label_3class", "count"])
        writer.writeheader()
        for label, count in counts.items():
            writer.writerow({"label_3class": label, "count": count})
    print(counts)


if __name__ == "__main__":
    main()
