import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.utils import to_categorical

# Load the saved model
model_path = "/Users/nikitanair/PycharmProjects/pythonProject3/Model2/keras_model.h5"
model = load_model(model_path)

# Recompile the model manually
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss=CategoricalCrossentropy(),
    metrics=["accuracy"]
)

# Define constants
img_size = 300
labels = ["down", "fist", "okay", "palm", "thumbs", "up"]
num_classes = len(labels)

# Generate Dummy Test Data (Replace with your actual test data)
X_test = np.random.rand(100, img_size, img_size, 1)  # Replace with actual test images
y_test = np.random.randint(0, num_classes, 100)  # Replace with actual test labels
y_test = to_categorical(y_test, num_classes=num_classes)  # One-hot encode the labels

# Evaluate the model on the test data
results = model.evaluate(X_test, y_test, verbose=1)
print(f"Test Loss: {results[0]}")
print(f"Test Accuracy: {results[1]}")

# Simulated Historical Data (Replace with actual training history if available)
epochs = 10
train_accuracy = [0.6 + i * 0.03 for i in range(epochs)]  # Simulated training accuracy
val_accuracy = [0.55 + i * 0.025 for i in range(epochs)]  # Simulated validation accuracy

# Plot the accuracy graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), train_accuracy, label="Training Accuracy")
plt.plot(range(1, epochs + 1), val_accuracy, label="Validation Accuracy", linestyle="--")
plt.title("Model Accuracy Over Epochs")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.grid()
plt.show()
