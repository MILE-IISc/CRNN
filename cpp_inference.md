
## [Convert .HDF5 to .pb and .onnx](https://stackoverflow.com/a/53386325/2969390)

### Convert .HDF5 to .pb
Add following line at the end of `eval.py`

```
model.save('./result/kan180k/')  # SavedModel format
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
model = keras.models.load_model('./result/kan180k/', compile=False)
```

[UserWarning: No training configuration found in save file: the model was *not* compiled. Compile it manually](https://stackoverflow.com/questions/53295570/userwarning-no-training-configuration-found-in-save-file-the-model-was-not-c)

### Convert .pb to .onxx

```
$ pip install onnxruntime
$ pip install tf2onnx
$ python -m tf2onnx.convert --saved-model ./result/kan180k/ --opset 13 --output ./result/kan180k.onnx
```

## [Tensorflow2 model inference in C++](https://medium.com/analytics-vidhya/inference-tensorflow2-model-in-c-aa73a6af41cf)

## [TensorFlow for Java](https://github.com/tensorflow/java)

* [Install TensorFlow Java](https://www.tensorflow.org/jvm/install)

* [Introduction to Tensorflow for Java](https://www.baeldung.com/tensorflow-java)

* [JavaDocs - Tensorflow for Java](https://www.tensorflow.org/jvm/api_docs/java/org/tensorflow/package-summary)
