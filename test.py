import numpy as np
import tensorflow as tf
import keras

# Load the VGG16 model with pre-trained ImageNet weights
vgg16 = keras.applications.VGG16(weights="imagenet")

# Create a converter for the VGG16 model
converter = tf.lite.TFLiteConverter.from_keras_model(vgg16)

# Optional: Apply optimizations to reduce model size
# This step can include quantization or other optimization techniques
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # Default optimization includes quantization

# Convert the model to TFLite format
tflite_model = converter.convert()

# Save the TFLite model to a file
with open("examples\\vgg16\\vgg16_model.tflite", "wb") as f:
    f.write(tflite_model)

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

# Get the input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create dummy input data based on the expected input shape
import numpy as np

input_shape = input_details[0]['shape']
input_data = np.random.rand(*input_shape).astype(np.float32)

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

print("Inference output:", output_data)
