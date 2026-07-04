from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from .config import DEFAULT_THRESHOLD, INFER_BATCH_SIZE, OVERLAP, TILE_SIZE
from .inference import OnnxSegmentationModel, infer_image_probability, make_overlay


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--tile_size", type=int, default=TILE_SIZE)
    parser.add_argument("--overlap", type=int, default=OVERLAP)
    parser.add_argument("--batch_size", type=int, default=INFER_BATCH_SIZE)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    model = OnnxSegmentationModel(args.model) if args.model else OnnxSegmentationModel()
    probability, meta = infer_image_probability(
        args.image,
        model,
        tile_size=args.tile_size,
        overlap=args.overlap,
        batch_size=args.batch_size,
    )
    mask = probability >= args.threshold
    prob_path = args.out / "probability.npy"
    mask_path = args.out / "mask.png"
    overlay_path = args.out / "overlay_preview.jpg"
    meta_path = args.out / "inference_meta.json"
    import numpy as np

    np.save(prob_path, probability)
    cv2.imwrite(str(mask_path), mask.astype("uint8") * 255)
    make_overlay(args.image, mask, overlay_path)
    meta["threshold"] = args.threshold
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(args.out)


if __name__ == "__main__":
    main()

