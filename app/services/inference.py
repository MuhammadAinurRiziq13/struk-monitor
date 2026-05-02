import os
import numpy as np
import tensorflow as tf
import keras

# Kita biarkan patch tetap ada kalau suatu saat mau balik ke H5
def patched_init(original_init):
    def new_init(self, *args, **kwargs):
        if 'batch_shape' in kwargs and 'shape' not in kwargs:
            bs = kwargs['batch_shape']
            if isinstance(bs, list) and len(bs) > 1:
                kwargs['shape'] = tuple(bs[1:])
        kwargs.pop('quantization_config', None)
        kwargs.pop('batch_shape', None)
        kwargs.pop('optional', None)
        kwargs.pop('dtype_policy', None)
        original_init(self, *args, **kwargs)
    return new_init

keras.layers.Dense.__init__ = patched_init(keras.layers.Dense.__init__)
keras.layers.LSTM.__init__ = patched_init(keras.layers.LSTM.__init__)
keras.layers.InputLayer.__init__ = patched_init(keras.layers.InputLayer.__init__)

# Path model TFLite
TFLITE_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/model.tflite')
LABEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/labels.npy')

def load_model_and_labels():
    """Memuat model TFLite dan label."""
    try:
        # Load labels
        labels = np.load(LABEL_PATH)
        print(f"[INFO] Label dimuat: {labels}")

        # Load TFLite Model
        print(f"[INFO] Memuat model TFLite: {TFLITE_MODEL_PATH}")
        # Gunakan interpreter standar
        interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        return interpreter, input_details, output_details, labels
    except Exception as e:
        print(f"[ERROR] Gagal memuat model TFLite: {e}")
        return None, None, None, []

def predict(interpreter, input_details, output_details, buffer):
    """Melakukan prediksi menggunakan model TFLite."""
    if interpreter is None:
        return np.zeros(1)

    # Preprocessing: Shape (1, 30, 63)
    input_data = np.array([buffer], dtype=np.float32)
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]
