import os
import numpy as np
import tensorflow as tf

# Path model TFLite
TFLITE_MODEL_PATH = os.path.join(os.path.dirname(__file__), '../../models/model.tflite')
LABEL_PATH        = os.path.join(os.path.dirname(__file__), '../../models/labels.npy')


def load_model_and_labels():
    """Memuat model TFLite dan label."""
    try:
        labels = np.load(LABEL_PATH)
        print(f"[INFO] Label dimuat: {labels}")

        print(f"[INFO] Memuat model TFLite: {TFLITE_MODEL_PATH}")
        interpreter = tf.lite.Interpreter(
            model_path=TFLITE_MODEL_PATH,
            num_threads=4,  # Manfaatkan multi-core CPU
        )
        interpreter.allocate_tensors()

        input_details  = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        return interpreter, input_details, output_details, labels

    except Exception as e:
        print(f"[ERROR] Gagal memuat model TFLite: {e}")
        return None, None, None, []


def predict(interpreter, input_details, output_details, buffer):
    """Melakukan prediksi menggunakan model TFLite."""
    if interpreter is None:
        return np.zeros(1)

    # Shape (1, 30, 63)
    input_data = np.array([buffer], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])[0]
