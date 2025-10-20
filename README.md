# Dangerous-Farm-Insects-Classification

This repository contains code for classifying dangerous farm insects. The project expects the dataset to be provided by the team rather than included in the repository.

Dataset placement
-----------------

Please place the dataset folder named "farm_insects" inside the `data/raw/` directory at the repository root. The expected path is:

```
data/raw/farm_insects/
```

Example structure (top-level of the dataset folder):

```
data/raw/farm_insects/
  Africanized Honey Bees (Killer Bees)/
  Aphids/
  Armyworms/
  ...
```

We intentionally ignore `data/` in version control to avoid committing large files. After placing the dataset, run the preprocessing script at `src/main.py` to prepare data for training.

- Code entry point: `src/main.py`
- Data preprocessing script: `src/data/data_preprocessor.py`
- Data paths configuration: `src/config/data_paths.py`

Using paths in code
-------------------

When developing CNN architectures or scripts you can import dataset paths directly from `src/config/data_paths.py`. This file exposes convenient Path objects such as `TRAINING_SET`, `VALIDATION_SET`, and `TESTING_SET` so your training code can use the correct filesystem locations without hardcoding paths.