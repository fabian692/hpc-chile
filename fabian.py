import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np
import matplotlib.pyplot as plt




# Listar las GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Limitar la visibilidad solo a la GPU 6
        tf.config.set_visible_devices(gpus[6], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[6], True)
        print("Usando GPU:", gpus[6])
    except RuntimeError as e:
        print(e)
# Optional: Force CPU to bypass GPU issues (temporary)
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

# 1. Load CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# 2. Preprocess data
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Class names for CIFAR-10
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# 3. Create CNN model
model = keras.Sequential([
    layers.Input(shape=(32, 32, 3)),  # Fix for Keras warning
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
    layers.BatchNormalization(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(10, activation='softmax')
])

# 4. Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 5. Model summary
model.summary()

# 6. Define early stopping
early_stopping = EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

# 7. Train the model
history = model.fit(
    x_train, y_train,
    epochs=100,
    validation_data=(x_test, y_test),
    batch_size=64,
    callbacks=[early_stopping]
)

# 8. Evaluate the model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc:.4f}')

# 9. Visualize training results
plt.figure(figsize=(12, 4))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy During Training')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss During Training')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.savefig('training_results.png')
plt.show()

# 10. Show some predictions
predictions = model.predict(x_test[:5])
for i in range(5):
    predicted_label = np.argmax(predictions[i])
    true_label = y_test[i][0]
    print(f'Image {i+1}: Predicted: {class_names[predicted_label]}, Actual: {class_names[true_label]}')
