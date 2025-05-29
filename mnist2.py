import tensorflow as tf
import numpy as np

# Verificar GPUs disponibles
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs detectadas: {[gpu.name for gpu in gpus]}")

# Asegúrate de tener al menos 7 GPUs (índices 0 a 6)
if len(gpus) < 7:
    raise RuntimeError("Este ejemplo requiere al menos 7 GPUs (para usar la 5 y 6)")

# Limitar visibilidad solo a GPU 5 y GPU 6
tf.config.set_visible_devices([gpus[1], gpus[2]], 'GPU')

# Habilitar crecimiento dinámico de memoria
for gpu in [gpus[5], gpus[6]]:
    tf.config.experimental.set_memory_growth(gpu, True)

# Crear estrategia distribuida con GPU 5 y 6
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:1", "/gpu:2"])
print(f"Usando estrategia con {strategy.num_replicas_in_sync} GPUs")

# Cargar dataset MNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Preprocesamiento
x_train = x_train.astype('float32') / 255.0
x_test  = x_test.astype('float32') / 255.0
x_train = np.expand_dims(x_train, -1)  # Añadir canal
x_test  = np.expand_dims(x_test, -1)

# Crear modelo dentro del scope
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

# Entrenar
model.fit(x_train, y_train, epochs=5, batch_size=256, validation_data=(x_test, y_test))
