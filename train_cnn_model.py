# train_cnn_model.py
import os
import pickle

import numpy as np
from keras.layers import (BatchNormalization, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tqdm.keras import TqdmCallback


def load_data():
    """Memuat data wajah dan label dari faces_data.pkl."""
    if os.path.exists('data/faces_data.pkl'):
        with open('data/faces_data.pkl', 'rb') as f:
            faces_data = pickle.load(f)
        return faces_data
    else:
        print("File data/faces_data.pkl tidak ditemukan.")
        return None

def prepare_data(faces_data, validation_split=True):
    """Memproses data wajah dan label agar siap digunakan untuk pelatihan CNN."""
    # Menggabungkan semua wajah menjadi satu array besar
    X = np.vstack(faces_data)
    
    # Membuat label untuk setiap set wajah berdasarkan indeks urutannya
    y = np.array([i for i, face_set in enumerate(faces_data) for _ in range(len(face_set))])
    
    # Mengubah bentuk X agar sesuai dengan input CNN
    X = X.reshape(X.shape[0], 64, 64, 1).astype('float32') / 255.0
    
    if validation_split:
        # Membagi data menjadi set pelatihan dan validasi
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_val, y_train, y_val
    else:
        return X, y  # Mengembalikan hanya X dan y jika validation_split=False

def build_model(num_classes):
    """Membangun model CNN untuk klasifikasi wajah dengan regularisasi dan batch normalization."""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.4),

        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_and_save_model(X_train, y_train, X_val, y_val, num_classes, epochs=20, batch_size=32):
    """Melatih model CNN dengan data augmentation dan menyimpannya dalam file .h5."""
    model = build_model(num_classes)
    print("Mulai pelatihan model CNN...")

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True
    )
    datagen.fit(X_train)

    # Pelatihan model dengan progress bar dari Tqdm
    model.fit(datagen.flow(X_train, y_train, batch_size=batch_size),
              validation_data=(X_val, y_val) if X_val is not None else None,
              epochs=epochs,
              callbacks=[TqdmCallback()],
              steps_per_epoch=len(X_train) // batch_size)

    print("Pelatihan selesai. Menyimpan model...")

    # Simpan model yang sudah terlatih
    if not os.path.exists('data'):
        os.makedirs('data')
    model.save('data/face_recognition_cnn.h5')
    print("Model berhasil disimpan sebagai 'data/face_recognition_cnn.h5'.")

def main():
    # Muat data wajah
    faces_data = load_data()
    if faces_data is None or len(faces_data) == 0:
        print("Tidak ada data untuk dilatih.")
        return
    
    # Siapkan data untuk pelatihan
    X_train, X_val, y_train, y_val = prepare_data(faces_data)
    num_classes = len(faces_data)  # Jumlah kelas sesuai dengan jumlah pengguna yang berbeda

    # Latih dan simpan model
    train_and_save_model(X_train, y_train, X_val, y_val, num_classes)

if __name__ == '__main__':
    main()