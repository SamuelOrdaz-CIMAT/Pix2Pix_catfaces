"""Generate Pix2Pix dataset from CatFLW.

A: single-channel gaussian landmark map saved as .npy
B: RGB cat crop (with background).
"""

import os
import argparse
import json
import random
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt


def create_heatmap_landmarks(coords_scaled, size=256, sigma=4):
    """Create a single-channel gaussian heatmap (uint8) from landmarks."""
    heatmap = np.zeros((size, size), dtype=np.float32)
    if coords_scaled is None or len(coords_scaled) == 0:
        return heatmap.astype(np.uint8)

    two_sigma_sq = 2 * (sigma ** 2) + 1e-8
    win = int(max(2, sigma * 3))

    for x, y in coords_scaled:
        if not (0 <= x < size and 0 <= y < size):
            continue
        x0 = max(0, int(x - win))
        x1 = min(size, int(x + win + 1))
        y0 = max(0, int(y - win))
        y1 = min(size, int(y + win + 1))
        xs = np.arange(x0, x1)
        ys = np.arange(y0, y1)
        xx, yy = np.meshgrid(xs, ys)
        dist_sq = (xx - x) ** 2 + (yy - y) ** 2
        heatmap[y0:y1, x0:x1] += np.exp(-dist_sq / two_sigma_sq)

    m = heatmap.max()
    if m > 0:
        heatmap = (heatmap / m) * 255.0
    return heatmap.astype(np.uint8)


def adapt_bbox(coords, w, h, given_bbox=None, init_margin_factor=0.5, min_coverage=0.9, expand_step=0.2, max_expansions=5, min_bbox_side=80):
    """Adaptive bbox expansion to reach landmark coverage target.

    Returns (x_min, y_min, x_max, y_max, coverage, used_fallback).
    """
    coords = np.asarray(coords, dtype=np.float32)
    used_fallback = False

    if given_bbox is not None and len(given_bbox) >= 4:
        x_min, y_min, x_max, y_max = [int(v) for v in given_bbox[:4]]
    else:
        lm_x_min, lm_y_min = coords[:, 0].min(), coords[:, 1].min()
        lm_x_max, lm_y_max = coords[:, 0].max(), coords[:, 1].max()
        lm_w = lm_x_max - lm_x_min
        lm_h = lm_y_max - lm_y_min
        margin_x = max(30, int(lm_w * init_margin_factor))
        margin_y = max(30, int(lm_h * init_margin_factor))
        x_min = int(max(0, lm_x_min - margin_x))
        y_min = int(max(0, lm_y_min - margin_y))
        x_max = int(min(w, lm_x_max + margin_x))
        y_max = int(min(h, lm_y_max + margin_y))

    def coverage(x0, y0, x1, y1):
        inside = ((coords[:, 0] >= x0) & (coords[:, 0] <= x1) & (coords[:, 1] >= y0) & (coords[:, 1] <= y1)).sum()
        return inside / float(len(coords) + 1e-6)

    cov = coverage(x_min, y_min, x_max, y_max)
    expansions = 0
    while cov < min_coverage and expansions < max_expansions:
        expand_x = int((x_max - x_min) * expand_step)
        expand_y = int((y_max - y_min) * expand_step)
        x_min = max(0, x_min - expand_x)
        y_min = max(0, y_min - expand_y)
        x_max = min(w, x_max + expand_x)
        y_max = min(h, y_max + expand_y)
        cov = coverage(x_min, y_min, x_max, y_max)
        expansions += 1

    bw, bh = (x_max - x_min), (y_max - y_min)
    if bw < min_bbox_side or bh < min_bbox_side or cov < min_coverage * 0.6:
        x_min, y_min, x_max, y_max = 0, 0, w, h
        cov = coverage(x_min, y_min, x_max, y_max)
        used_fallback = True

    return x_min, y_min, x_max, y_max, cov, used_fallback


def prepare_catflw_dataset(input_root, output_root, sigma=4, min_coverage=0.9, expand_step=0.25, max_expansions=6, min_bbox_side=80):
    """Convert CatFLW to Pix2Pix format (1-channel landmarks + RGB image)."""
    if not os.path.exists(input_root):
        print(f"Input root not found: {input_root}")
        return

    os.makedirs(output_root, exist_ok=True)
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(output_root, split, "A"), exist_ok=True)
        os.makedirs(os.path.join(output_root, split, "B"), exist_ok=True)

    img_dir = os.path.join(input_root, "images")
    label_dir = os.path.join(input_root, "labels")
    
    if not os.path.isdir(img_dir) or not os.path.isdir(label_dir):
        print("Expected structure: <input_root>/images and <input_root>/labels")
        return

    all_files = [f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".png"))]
    random.shuffle(all_files)
    n_total = len(all_files)
    
    if n_total == 0:
        print("No images found.")
        return

    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)
    
    splits = {
        "train": all_files[:n_train],
        "val": all_files[n_train:n_train + n_val],
        "test": all_files[n_train + n_val:]
    }

    print(f"Total: {n_total} | Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
    
    skipped, processed = 0, 0

    for split, files in splits.items():
        print(f"Processing {split}...")
        
        for fname in tqdm(files, desc=split):
            img_path = os.path.join(img_dir, fname)
            base_name = os.path.splitext(fname)[0]
            json_path = os.path.join(label_dir, base_name + ".json")

            if not os.path.exists(json_path):
                skipped += 1
                continue

            img = cv2.imread(img_path)
            if img is None:
                skipped += 1
                continue

            try:
                with open(json_path, "r") as f:
                    data = json.load(f)
            except Exception:
                skipped += 1
                continue

            coords = None
            bbox = None
            if isinstance(data, dict):
                coords = data.get("labels") or data.get("landmarks") or data.get("points")
                bbox = data.get("bounding_boxes") or data.get("bbox") or data.get("box")
            
            if coords is None:
                skipped += 1
                continue
            
            coords = np.array(coords, dtype=np.float32)
            if coords.ndim != 2 or coords.shape[1] < 2:
                skipped += 1
                continue

            h, w = img.shape[:2]
            given_bbox = None
            if bbox is not None:
                try:
                    arr = np.array(bbox).flatten()
                    if arr.size >= 4:
                        given_bbox = arr[:4]
                except Exception:
                    pass

            x_min, y_min, x_max, y_max, cov, used_fallback = adapt_bbox(
                coords, w, h, given_bbox=given_bbox, init_margin_factor=0.5,
                min_coverage=min_coverage, expand_step=expand_step,
                max_expansions=max_expansions, min_bbox_side=min_bbox_side
            )

            if x_max <= x_min + 1 or y_max <= y_min + 1:
                skipped += 1
                continue

            cropped = img[y_min:y_max, x_min:x_max]
            if cropped.size == 0:
                skipped += 1
                continue
                
            ch, cw = cropped.shape[:2]
            scale_x, scale_y = 256.0 / cw, 256.0 / ch
            
            coords_scaled = coords.copy()
            coords_scaled[:, 0] = (coords_scaled[:, 0] - x_min) * scale_x
            coords_scaled[:, 1] = (coords_scaled[:, 1] - y_min) * scale_y

            img_resized = cv2.resize(cropped, (256, 256))
            mask = create_heatmap_landmarks(coords_scaled, size=256, sigma=sigma)
            pil_img = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
            
            outA = os.path.join(output_root, split, "A", base_name + ".npy")
            outB = os.path.join(output_root, split, "B", base_name + ".jpg")

            try:
                np.save(outA, mask)
                pil_img.save(outB, quality=95)
                processed += 1
            except Exception:
                skipped += 1

    if skipped > 0:
        print(f"Skipped: {skipped}")
    generate_preview(output_root, sigma)


def generate_preview(output_root, sigma):
    """Generate a simple dataset preview image."""
    try:
        folderA = os.path.join(output_root, "train", "A")
        folderB = os.path.join(output_root, "train", "B")
        
        filesA = [f for f in os.listdir(folderA) if f.endswith('.npy')]
        if not filesA:
            return

        example_file = random.choice(filesA)
        mask = np.load(os.path.join(folderA, example_file))
        
        img_name = example_file.replace('.npy', '.jpg')
        if not os.path.exists(os.path.join(folderB, img_name)):
            img_name = example_file.replace('.npy', '.png')
        
        real_img = Image.open(os.path.join(folderB, img_name)).convert("RGB")

        fig, axes = plt.subplots(1, 3, figsize=(10, 4))
        
        axes[0].imshow(mask, cmap='hot')
        axes[0].set_title('Landmarks heatmap')
        axes[0].axis('off')

        axes[1].imshow(real_img)
        axes[1].set_title('Real B')
        axes[1].axis('off')

        axes[2].imshow(real_img)
        axes[2].imshow(mask, cmap='hot', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.suptitle(f'Single-channel landmarks (sigma={sigma})', fontsize=12)
        plt.tight_layout()
        
        preview_path = os.path.join(output_root, "dataset_preview.png")
        plt.savefig(preview_path, bbox_inches='tight', dpi=100)
        plt.close()
        
    except Exception as e:
        print(f"Preview failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Generate Pix2Pix dataset from CatFLW (1ch heatmap + RGB)')
    parser.add_argument('--input_root', default='CatFLW dataset', help='Root folder with images/ and labels/')
    parser.add_argument('--output_root', default='datasets/catflw', help='Destination root')
    parser.add_argument('--sigma', type=float, default=2.0, help='Gaussian sigma (2-4 recommended)')
    parser.add_argument('--min_coverage', type=float, default=0.9, help='Target landmark coverage inside bbox')
    parser.add_argument('--expand_step', type=float, default=0.25, help='Relative expansion per attempt')
    parser.add_argument('--max_expansions', type=int, default=6, help='Max expansion iterations')
    parser.add_argument('--min_bbox_side', type=int, default=80, help='Minimum bbox side length before fallback')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--no_preview', action='store_true', help='Skip preview generation')
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Generating dataset...")
    print(f"Input: {args.input_root}")
    print(f"Output: {args.output_root}")
    print(f"Sigma: {args.sigma} | Coverage: {args.min_coverage} | Seed: {args.seed}")
    
    prepare_catflw_dataset(
        args.input_root,
        args.output_root,
        sigma=args.sigma,
        min_coverage=args.min_coverage,
        expand_step=args.expand_step,
        max_expansions=args.max_expansions,
        min_bbox_side=args.min_bbox_side,
    )

    print("Dataset generation complete.")


if __name__ == "__main__":
    main()