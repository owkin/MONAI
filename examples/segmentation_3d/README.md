# Usage

## Local testing in python env

```sh
pip install -r scripts/run_local_requirements.txt
python scripts/generate_data_samples.py
python scripts/run_local.py
```

## Local testing in docker env

```sh
substra run-local \
    assets/algo \
    --train-opener assets/train_dataset/opener.py \
    --test-opener assets/test_dataset/opener.py \
    --metrics assets/objective \
    --train-data-samples assets/train_data_samples \
    --test-data-samples assets/test_data_samples
```

# Caveats

data sample files have their named stored in metadata. And since these metadata are pushed to the save_prediction,
if all files have the same name (just in different folders) the predictions will also all have the same name, overwriting each other.

NiftiSaver uses the channel last format, which means we have to revert to channel first when loading saved predictions

Code from NiftiSaver:

```python
# change data to "channel last" format and write to nifti format file
data = np.moveaxis(data, 0, -1)
```