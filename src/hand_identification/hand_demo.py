import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mediapipe as mp

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "hand_landmarker.task"

FINGER_PATHS = [
    [0, 1, 2, 3, 4],      # thumb
    [0, 5, 6, 7, 8],      # index
    [9, 10, 11, 12],      # middle
    [13, 14, 15, 16],     # ring
    [0, 17, 18, 19, 20],  # pinky
    [5, 9, 13, 17],       # base of the fingers
]

def main() -> None:
    parser = argparse.ArgumentParser(description="Detecting 21 hand's landmarks")
    parser.add_argument("image", type=Path, help="path to image")
    parser.add_argument("--hands", type=int, choices=(1, 2), default=1)
    parser.add_argument("--save", action="store_true", help="save image to results/")
    args = parser.parse_args()

    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    if not MODEL_PATH.is_file():
        parser.error(f"Model not found: {MODEL_PATH}")

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=args.hands,
    )

    image = mp.Image.create_from_file(str(args.image))
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        result = detector.detect(image)

    rgb = image.numpy_view()
    height, width = rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(rgb)
    ax.axis("off")

    for index, hand in enumerate(result.hand_landmarks):
        points = [(p.x * width, p.y * height) for p in hand]

        for path in FINGER_PATHS:
            ax.plot([points[i][0] for i in path], [points[i][1] for i in path], linewidth=2)

        ax.scatter(*zip(*points), color="white", edgecolor="black", s=28, zorder=3)
        for number, (x, y) in enumerate(points):
            ax.annotate(str(number), (x, y), xytext=(4, 4),
                        textcoords="offset points", color="yellow", fontsize=8)

        side = result.handedness[index][0].category_name
        print(f"Hand {index + 1}: {side}, detected {len(hand)} points")

    if not result.hand_landmarks:
        print("Hand not found.")

    if args.save:
        output_dir = PROJECT_ROOT / "results"
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{args.image.stem}_skeleton.png"
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {output_path}")

    plt.show()


if __name__ == "__main__":
    main()
