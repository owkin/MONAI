# Usage

```sh
pip install -r scripts/run_local_requirements.txt
python scripts/generate_data_samples.py
python scripts/run_local.py
```

# Caveats

* data sample files have their named stored in metadata. And since these metadata are pushed to the save_prediction,
  if all files have the same name (just in different folders) the predictions will also all have the same name, overwriting each other.