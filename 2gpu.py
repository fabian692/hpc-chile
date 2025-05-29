import tensorflow as tf
import numpy as np

# Verifica que hay al menos 7 GPUs (de 0 a 6)
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs disponibles: {[gpu.name for gpu in gpus]}")

# Establecer visibilidad solo para GPU 5 y 6
tf.config.set_visible_devices([gpus[5], gpus[6]], 'GPU')

# Habilitar crecimiento de memoria opcionalmente
for gpu in [gpus[1], gpus[2]]:
    tf.config.experimental.set_memory_growth(gpu, True)

# Crear la estrategia para usar GPU 5 y 6
strategy = tf.distribute.MirroredStrategy(devices=["/gpu:1", "/gpu:2"])
print(f"Usando estrategia con {strategy.num_replicas_in_sync} GPUs")

# Datos de ejemplo
x_train = np.random.rand(10000, 32)
y_train = np.random.randint(0, 2, size=(10000, 1))

# Definir el modelo dentro del scope de la estrategia
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

# Entrenamiento
model.fit(x_train, y_train, batch_size=256, epochs=5)

