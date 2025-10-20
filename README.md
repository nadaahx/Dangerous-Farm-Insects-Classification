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

Development Guidelines
--------------------

### CNN Architectures

Here's a suggested approach for organizing CNN architectures in this project. Feel free to propose and implement alternative structures if they better suit your needs:

1. Consider creating a new Python file in the `src/architectures/` directory
2. You might want to name the file after your architecture (e.g., `alexnet.py`, `lenet.py`)
3. One recommended pattern is implementing your model as a class that can be imported from the main file

Suggested structure (but open to better alternatives):

```
src/
  architectures/
    alexnet.py      # Contains AlexNetModel class
    lenet.py        # Contains LeNetModel class
    vgg.py          # Contains VGGModel class
  main.py          # Imports and uses the models
```

This is just one way to organize the code - if you have ideas for a better structure or a different approach that could benefit the project, please feel free to suggest and implement it. The goal is to maintain clean, maintainable code while allowing flexibility for different architectural approaches.