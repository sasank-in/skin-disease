import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Batch predict images with YOLOv8 classifier.")
    parser.add_argument(
        "--model",
        default="checkpoints/best.pt",
        help="Path to model weights.",
    )
    parser.add_argument(
        "--input",
        default="input_images",
        help="Folder with input images.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=3,
        help="Number of top predictions to display.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model)
    input_dir = Path(args.input)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    model = YOLO(str(model_path))

    images = sorted(
        [p for p in input_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}]
    )
    if not images:
        print("No images found.")
        return

    for img in images:
        res = model(str(img), verbose=False)[0]
        names = model.names
        top_k = max(1, min(args.topk, len(names)))

        if top_k <= 5:
            top_indices = list(res.probs.top5)[:top_k]
            top_scores = list(res.probs.top5conf)[:top_k]
        else:
            arr = res.probs.data
            try:
                import numpy as np

                arr = arr.detach().cpu().numpy() if hasattr(arr, "detach") else np.array(arr)
                top_indices = np.argsort(arr)[::-1][:top_k].tolist()
                top_scores = arr[top_indices].tolist()
            except Exception:
                top_indices = list(res.probs.top5)[:5]
                top_scores = list(res.probs.top5conf)[:5]
        print(f"\n{img.name}")
        for idx, score in zip(top_indices, top_scores):
            label = names[int(idx)]
            print(f"  {label:<20} {float(score)*100:6.2f}%")


if __name__ == "__main__":
    main()
