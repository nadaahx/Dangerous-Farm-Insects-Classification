from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"

RAW_DATA_DIR = DATA_DIR / "raw" / "farm_insects"
PROCESSED_DATA_DIR = DATA_DIR / "processed" / "farm_insects"
SPLITS_DIR = PROCESSED_DATA_DIR / "splits"

AUGMENTED_TRAINING_SPLIT = PROCESSED_DATA_DIR / "augmented_train"
TRAINING_SPLIT = SPLITS_DIR / "train"


TRAINING_SET = (
    AUGMENTED_TRAINING_SPLIT if AUGMENTED_TRAINING_SPLIT.exists() else TRAINING_SPLIT
)
VALIDATION_SET = SPLITS_DIR / "val"
TESTING_SET = SPLITS_DIR / "test"
