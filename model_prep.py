import tensorflow as tf


path = r"C:\Users\jonoj\Documents\ML\Models\absBest\trainedModel"
converter = tf.lite.TFLiteConverter.from_saved_model(path)
liteModel = converter.convert()

with open(r"assets\model.tflite", "wb") as f:
    f.write(liteModel)
