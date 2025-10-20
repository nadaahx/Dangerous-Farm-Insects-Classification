import shutil
import numpy as np
from PIL import Image
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
    img_to_array,
)


class Data_Preprocessor:
    """Preprocesses image datasets: splits data and balances classes via augmentation."""

    def __init__(self, source_dir, output_dir, n_splits=5, balance_threshold=0.7):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.n_splits = n_splits
        self.balance_threshold = balance_threshold

    def run(self, fold_idx=0, fresh_start=True):
        """Run complete preprocessing: split data, then check balance and augment if needed."""
        if self.output_dir.exists():
            if fresh_start:
                print(f"Removing existing output directory: {self.output_dir}")
                shutil.rmtree(self.output_dir)
            else:
                print(f"Output directory already exists: {self.output_dir}")
                print("Skipping preprocessing. Set fresh_start=True to reprocess.")
                return
        self._split_data(fold_idx)
        self._balance_and_augment()

    def _split_data(self, fold_idx):
        """Split dataset using stratified k-fold cross-validation."""
        print(f"Splitting data (fold {fold_idx} as test)...")

        unique_files = {}

        for class_dir in sorted(self.source_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                for file_path in class_dir.iterdir():
                    if file_path.is_file() and file_path.suffix.lower() in [
                        ".jpg",
                        ".jpeg",
                        ".png",
                    ]:
                        resolved = file_path.resolve()
                        unique_files[str(resolved)] = (str(file_path), class_name)

        paths = []
        labels = []
        for resolved_path, (original_path, class_name) in sorted(unique_files.items()):
            paths.append(original_path)
            labels.append(class_name)

        X, y = np.array(paths), np.array(labels)
        print(f"Found {len(X)} unique images from {len(set(y))} classes")

        class_counts = {}
        for label in set(y):
            class_counts[label] = np.sum(y == label)
        print("\nImages per class in source:")
        for class_name in sorted(class_counts.keys()):
            print(f"  {class_name}: {class_counts[class_name]}")

        # Create test split
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        splits = list(skf.split(X, y))
        test_idx = splits[fold_idx][1]
        remaining_idx = splits[fold_idx][0]

        # Create train/val split from remaining data
        X_rem, y_rem = X[remaining_idx], y[remaining_idx]
        skf_val = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, val_idx = next(skf_val.split(X_rem, y_rem))

        # Save splits
        self._save_images(X_rem[train_idx], y_rem[train_idx], "splits/train")
        self._save_images(X_rem[val_idx], y_rem[val_idx], "splits/val")
        self._save_images(X[test_idx], y[test_idx], "splits/test")

        print(
            f"\nSplit sizes: {len(train_idx)} train, {len(val_idx)} val, {len(test_idx)} test"
        )
        print(f"Total: {len(train_idx) + len(val_idx) + len(test_idx)}\n")

    def _save_images(self, paths, labels, subdir):
        """Copy images to organized directory structure."""
        for path, label in zip(paths, labels):
            dest_dir = self.output_dir / subdir / label
            dest_dir.mkdir(parents=True, exist_ok=True)

            source_file = Path(path)
            dest_file = dest_dir / source_file.name

            if dest_file.exists():
                counter = 1
                stem = source_file.stem
                suffix = source_file.suffix
                while dest_file.exists():
                    dest_file = dest_dir / f"{stem}_{counter}{suffix}"
                    counter += 1
                print(
                    f"Warning: Renamed {source_file.name} to {dest_file.name} to avoid collision"
                )

            shutil.copy2(path, dest_file)

    def _balance_and_augment(self):
        """Check class balance and augment minority classes if needed."""
        print("Checking class balance...")
        train_dir = self.output_dir / "splits/train"

        # Count samples per class
        counts = {}
        for class_dir in sorted(train_dir.iterdir()):
            if class_dir.is_dir():
                # Count all image files
                img_files = [
                    f
                    for f in class_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
                ]
                counts[class_dir.name] = len(img_files)

        print("\nTraining set class distribution:")
        for name, count in sorted(counts.items()):
            print(f"  {name}: {count}")

        if not counts:
            print("No classes found in training set!")
            return

        min_count, max_count = min(counts.values()), max(counts.values())
        ratio = min_count / max_count
        print(f"\nBalance ratio: {ratio:.3f} (threshold: {self.balance_threshold})")

        if ratio >= self.balance_threshold:
            print("Dataset is balanced - no augmentation needed\n")
            return

        # Augment minority classes
        print("Dataset imbalanced - augmenting minority classes...")
        aug_dir = self.output_dir / "augmented_train"
        datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.15,
            height_shift_range=0.15,
            shear_range=0.15,
            zoom_range=0.3,
            horizontal_flip=True,
            vertical_flip=False,
            brightness_range=[0.7, 1.3],
            fill_mode="nearest",
            channel_shift_range=20.0,
        )

        for class_name, count in counts.items():
            class_dir = train_dir / class_name
            aug_class_dir = aug_dir / class_name
            aug_class_dir.mkdir(parents=True, exist_ok=True)

            images = [
                f
                for f in class_dir.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]

            for img in images:
                shutil.copy2(img, aug_class_dir / img.name)

            # Generate augmented images for minority classes
            if count < max_count:
                needed = max_count - count
                print(f"  {class_name}: generating {needed} augmented images")

                for i in range(needed):
                    src = images[i % len(images)]
                    img = img_to_array(load_img(src))
                    aug_img = datagen.flow(np.expand_dims(img, 0), batch_size=1)
                    augmented = next(aug_img)[0].astype("uint8")

                    # Save with unique name
                    aug_filename = aug_class_dir / f"aug_{i:04d}_{src.name}"
                    Image.fromarray(augmented).save(aug_filename)

        print(f"\nAugmentation complete!")
        print(f"Augmented dataset saved to: {aug_dir}\n")
