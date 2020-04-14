import importlib
import os

# Paths

current_dir = os.path.dirname(os.path.realpath(__file__))
assets_dir = os.path.realpath(os.path.join(current_dir, '../assets/'))

train_opener_path = os.path.join(assets_dir, 'train_dataset/opener.py')
test_opener_path = os.path.join(assets_dir, 'test_dataset/opener.py')
algo_path = os.path.join(assets_dir, 'algo/algo.py')
metrics_path = os.path.join(assets_dir, 'objective/metrics.py')

train_data_sample_folders = [
    os.path.join(assets_dir, 'train_data_samples', folder)
    for folder in os.listdir(os.path.join(assets_dir, 'train_data_samples'))
]

test_data_sample_folders = [
    os.path.join(assets_dir, 'test_data_samples', folder)
    for folder in os.listdir(os.path.join(assets_dir, 'test_data_samples'))
]

model_path = os.path.join(assets_dir, 'model')
predictions_path = os.path.join(assets_dir, 'predictions')

# Load modules

spec = importlib.util.spec_from_file_location('train_opener_module', train_opener_path)
train_opener_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_opener_module)

spec = importlib.util.spec_from_file_location('test_opener_module', test_opener_path)
test_opener_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_opener_module)

spec = importlib.util.spec_from_file_location('algo_module', algo_path)
algo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(algo_module)

spec = importlib.util.spec_from_file_location('metrics_module', metrics_path)
metrics_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics_module)

# Assets

train_opener = train_opener_module.MonaiTrainOpener()
test_opener = test_opener_module.MonaiTestOpener()
algo = algo_module.MonaiAlgo()

# Training

print('Loading train data')
X_train = train_opener.get_X(train_data_sample_folders)
y_train = train_opener.get_y(train_data_sample_folders)

print('Training model')
model = algo.train(X_train, y_train, None, 0)
print('Saving model')
algo.save_model(model, model_path)

# Predictions

print('Loading test data')
X_test = test_opener.get_X(test_data_sample_folders)
print('Loading model')
model = algo.load_model(model_path)

print('Predicting')
y_predictions = algo.predict(X_test, model)
print('Saving predictions')
test_opener.save_predictions(y_predictions, predictions_path)

print('Loading y_true')
y_true = test_opener.get_y(test_data_sample_folders)
print('Loading predictions')
y_pred = test_opener.get_predictions(predictions_path)

print('Calculating score')
metrics = metrics_module.MonaiMetrics()
score = metrics.score(y_true, y_pred)

print(f'Score: {score}')
