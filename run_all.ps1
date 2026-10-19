$ErrorActionPreference = "Stop"
python src/data/make_dataset.py
python src/models/train_model.py
python src/visualization/visualize.py
