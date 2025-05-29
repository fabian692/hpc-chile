import tensorflow as tf

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

# Tu código TensorFlow aquí
a = tf.random.normal([1000, 1000])
b = tf.random.normal([1000, 1000])
c = tf.matmul(a, b)

