import h5py
import numpy as np
from tensorflow.keras.models import load_model


def view_model_summary(model_path):
    # Muat model Keras dari file .h5
    try:
        model = load_model(model_path)
        print("Model Summary:")
        model.summary()  # Tampilkan ringkasan arsitektur model
    except Exception as e:
        print(f"Gagal memuat model: {e}")

def view_h5_weights(model_path):
    # Memuat file .h5 menggunakan h5py untuk melihat layer dan bobot
    try:
        with h5py.File(model_path, 'r') as file:
            print("\nDaftar Layer dalam .h5:")
            for layer in file['model_weights'].keys():
                print(f"Layer: {layer}")
                weights = file['model_weights'][layer]
                for weight in weights.keys():
                    print(f"  {weight}: shape = {weights[weight].shape}")
    except Exception as e:
        print(f"Gagal memuat bobot: {e}")

if __name__ == "__main__":
    model_path = 'data/face_recognition_cnn.h5'  # Ganti dengan lokasi file .h5 Anda
    view_model_summary(model_path)
    view_h5_weights(model_path)