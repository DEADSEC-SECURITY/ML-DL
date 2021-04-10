# Iris Plant Prediction With TensorFlow

## 📝 CONTRIBUTIONS

Before doing any contribution read <a href="https://github.com/DEADSEC-SECURITY/ML-DL/blob/main/CONTRIBUTING.md">CONTRIBUTING</a>.

## 📧 CONTACT

Email: amng835@gmail.com

General Discord: https://discord.gg/dFD5HHa

Developer Discord: https://discord.gg/rxNNHYN9EQ

## 📥 INSTALLING
```bash
pip install -r requirements.txt
```

## ⚙ HOW TO USE

Training the model:
```python
iris = IrisClass()
iris.train_model()
iris.plot_loss() -> Will display a graph with the loss
iris.plot_accuracy() -> Will display a graph with accuracy
iris.save_model() -> Save model to file
```
Make prediction:
```python
iris = IrisClass()
iris.load_model()
# Example data to predict
data = {
    'SepalLengthCm': [6, 5],
    'SepalWidthCm': [3, 2],
    'PetalLengthCm': [5, 4],
    'PetalWidthCm': [2, 1]
}
data = pd.DataFrame(data=data)
predictions = iris.make_prediction()
print(predictions) # This returns a 2D array of predictions and percentages
```

## 🤝 PARAMETERS
- models_dir : str, optional
  - Models directory
- models_file_name : str, optional
  - Model name
- logging : bool, optional 
  - Save logs or not
- data_csv : str, optional
  - Csv data path
- log_file : str, optional
  - Logs directory
