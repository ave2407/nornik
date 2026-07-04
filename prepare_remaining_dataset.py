from __future__ import annotations

import csv
import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path


SOURCE_ROOT = Path(
    r"C:\Users\Vladimir\Yandex.Disk\Загрузки\Задача 3. Скажи мне, кто твой шлиф"
)
OUTPUT_ROOT = Path("remaining_dataset_clean")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class SourceGroup:
    rel_dir: str
    target_group: str
    prefix: str


GROUPS = [
    SourceGroup("Фото руд по сортам. ч1/Оталькованные руды", "images_by_class/talc", "talc"),
    SourceGroup("Фото руд по сортам. ч2/оталькованные", "images_by_class/talc", "talc"),
    SourceGroup("Фото руд по сортам. ч1/Рядовые руды", "images_by_class/ordinary", "ordinary"),
    SourceGroup("Фото руд по сортам. ч2/рядовые", "images_by_class/ordinary", "ordinary"),
    SourceGroup("Фото руд по сортам. ч1/Труднообогатимые руды", "images_by_class/refractory", "refractory"),
    SourceGroup("Фото руд по сортам. ч2/тонкие", "images_by_class/thin", "thin"),
    SourceGroup("Панорамы", "panoramas", "panorama"),
    SourceGroup(
        "Фото руд по сортам. ч1/Оталькованные руды/Области оталькования",
        "annotations_blue_lines/talc_regions",
        "talc_region",
    ),
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def image_files(path: Path) -> list[Path]:
    return sorted(
        p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def reset_output_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def main() -> None:
    if not SOURCE_ROOT.exists():
        raise FileNotFoundError(SOURCE_ROOT)

    reset_output_dir(OUTPUT_ROOT)

    seen: dict[str, dict[str, str]] = {}
    mapping_rows: list[dict[str, str]] = []
    duplicate_rows: list[dict[str, str]] = []
    counters: dict[str, int] = {}

    for group in GROUPS:
        src_dir = SOURCE_ROOT / Path(group.rel_dir)
        if not src_dir.exists():
            raise FileNotFoundError(src_dir)

        dst_dir = OUTPUT_ROOT / group.target_group
        dst_dir.mkdir(parents=True, exist_ok=True)

        for src_path in image_files(src_dir):
            digest = sha256_file(src_path)
            source_rel = str(src_path.relative_to(SOURCE_ROOT))

            if digest in seen:
                duplicate_rows.append(
                    {
                        "sha256": digest,
                        "duplicate_source": source_rel,
                        "kept_source": seen[digest]["source_rel"],
                        "kept_target": seen[digest]["target_rel"],
                    }
                )
                continue

            counters[group.prefix] = counters.get(group.prefix, 0) + 1
            idx = counters[group.prefix]
            dst_name = f"{group.prefix}_{idx:05d}{src_path.suffix.lower()}"
            dst_path = dst_dir / dst_name
            shutil.copy2(src_path, dst_path)

            target_rel = str(dst_path.relative_to(OUTPUT_ROOT))
            row = {
                "sha256": digest,
                "source": source_rel,
                "target": target_rel,
                "group": group.target_group,
                "original_name": src_path.name,
            }
            mapping_rows.append(row)
            seen[digest] = {"source_rel": source_rel, "target_rel": target_rel}

    with (OUTPUT_ROOT / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["sha256", "source", "target", "group", "original_name"]
        )
        writer.writeheader()
        writer.writerows(mapping_rows)

    with (OUTPUT_ROOT / "duplicates_removed_sha256.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sha256", "duplicate_source", "kept_source", "kept_target"],
        )
        writer.writeheader()
        writer.writerows(duplicate_rows)

    summary: dict[str, int] = {}
    for row in mapping_rows:
        summary[row["group"]] = summary.get(row["group"], 0) + 1

    with (OUTPUT_ROOT / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "count"])
        writer.writeheader()
        for group, count in sorted(summary.items()):
            writer.writerow({"group": group, "count": count})

    print(f"output: {OUTPUT_ROOT.resolve()}")
    print(f"kept: {len(mapping_rows)}")
    print(f"duplicates_removed: {len(duplicate_rows)}")
    for group, count in sorted(summary.items()):
        print(f"{group}: {count}")


if __name__ == "__main__":
    main()
