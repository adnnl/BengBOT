import tensorflow as tf

model = tf.keras.models.load_model("model/inimodelygy.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# 🔥 tambahkan ini
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,
    tf.lite.OpsSet.SELECT_TF_OPS
]

converter._experimental_lower_tensor_list_ops = False

tflite_model = converter.convert()

with open("model/inimodelygy.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Convert selesai!")
