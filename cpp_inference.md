
## Convert .HDF5 to .pb and .onnx

* [How to export Keras .h5 to tensorflow .pb?](https://stackoverflow.com/a/53386325/2969390)

Add following line at the end of `eval.py`

```
model.save('./xor/')  # SavedModel format
```

```
$ pip install onnxruntime
$ pip install tf2onnx
$ python -m tf2onnx.convert --saved-model ./xor/ --opset 13 --output xor.onnx
```

