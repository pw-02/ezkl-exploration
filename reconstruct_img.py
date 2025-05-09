import json
import numpy as np
import matplotlib.pyplot as plt

# Load the JSON input (either from string or file)
with open("examples/onnx/mnist_classifier/input.json", "r") as f:
    data = json.load(f)

# Extract and reshape the input data
flat_image = np.array(data["input_data"]).squeeze()  # shape: (784,)
image = flat_image.reshape(28, 28)  # reshape to 28x28

# Display the image
plt.imshow(image, cmap="gray")
plt.title("Input Image")
plt.axis("off")
plt.show()
