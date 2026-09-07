from pathlib import Path
import random
import shutil

root = Path("datasets/coco128")
source_images = root / "images/train2017"
source_labels = root / "labels/train2017"
output = root.parent / "coco128_split"
random.seed(42)

images = sorted(source_images.glob("*.jpg"))
random.shuffle(images)
cut = int(len(images) * 0.8)
splits = {"train": images[:cut], "val": images[cut:]}

for split, files in splits.items():
    image_dir = output / "images" / split
    label_dir = output / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for image in files:
        shutil.copy2(image, image_dir / image.name)
        label = source_labels / f"{image.stem}.txt"
        if label.exists():
            shutil.copy2(label, label_dir / label.name)

print(f"train={len(splits['train'])}, val={len(splits['val'])}")
print(output)
