
## [Convert .HDF5 to .pb and .onnx](https://stackoverflow.com/a/53386325/2969390)

### Convert .HDF5 to .pb
Add following line at the end of `eval.py`

```
model.save('./xor/')  # SavedModel format
```

To test correctness of the SavedModel, revert above changes in `eval.py` and replace:

```
    _, model = CRNN(cfg)
    print (cfg.model_path);
    Model.load_weights(model, cfg.model_path)
```

with

```
from tensorflow import keras
...
model = keras.models.load_model('./xor/', compile=False)
```

### Convert .pb to .onxx

```
$ pip install onnxruntime
$ pip install tf2onnx
$ python -m tf2onnx.convert --saved-model ./xor/ --opset 13 --output xor.onnx
```

### [Tensorflow2 model inference in C++](https://medium.com/analytics-vidhya/inference-tensorflow2-model-in-c-aa73a6af41cf)

