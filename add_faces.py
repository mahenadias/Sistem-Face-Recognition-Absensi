import os
import pickle
import time  # Import modul time untuk menambahkan jeda

import cv2
import dlib
import numpy as np
import streamlit as st
from imutils import face_utils


def load_existing_data():
    if os.path.exists('data/faces_data.pkl') and os.path.exists('data/users.pkl'):
        with open('data/faces_data.pkl', 'rb') as f:
            faces_data = pickle.load(f)
        with open('data/users.pkl', 'rb') as f:
            users = pickle.load(f)
    else:
        faces_data = []  # Initialize as empty if no data found
        users = []
    return faces_data, users

def capture_face_data(cap, face_cascade, predictor, num_frames, instruction, wait_time=2):
    st.info(instruction)
    frames_collected = 0
    face_samples = []
    FRAME_WINDOW = st.image([])

    time.sleep(wait_time)  # Jeda sebelum memulai pengambilan frame
    while frames_collected < num_frames:
        ret, frame = cap.read()
        if not ret:
            st.error("Gagal membuka kamera.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            face = gray[y:y+h, x:x+w]
            resized_face = cv2.resize(face, (64, 64))
            face_samples.append(resized_face)
            frames_collected += 1

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"{frames_collected}/{num_frames}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        FRAME_WINDOW.image(frame, channels="BGR")

    return face_samples

def main():
    st.title("Tambah Wajah Baru ke Dataset")

    # Input user info
    name = st.text_input("Masukkan Nama:")
    user_id = st.text_input("Masukkan NIM:")
    major = st.text_input("Masukkan Prodi:")
    add_face_button = st.button("Tambah Wajah")

    if add_face_button and name and user_id and major:
        cap = cv2.VideoCapture(1)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")

        faces_data, users = load_existing_data()

        total_face_samples = []

        # Capture data for different orientations with a 3-second wait between orientations
        orientations = [
            "Tolong menghadap depan.",
            "Tolong menghadap kanan.",
            "Tolong menghadap kiri.",
            "Tolong menghadap bawah."
        ]
        for orientation in orientations:
            samples = capture_face_data(cap, face_cascade, predictor, 25, orientation, wait_time=3)
            total_face_samples.extend(samples)

        cap.release()
        st.success(f"100 gambar wajah berhasil disimpan untuk {name}.")

        if not os.path.exists('data'):
            os.makedirs('data')

        # Tambahkan wajah baru ke data yang sudah ada
        faces_data.append(np.array(total_face_samples))

        # Tambahkan user baru ke data pengguna yang sudah ada
        users.append({'name': name, 'user_id': user_id, 'major': major})

        with open('data/faces_data.pkl', 'wb') as f:
            pickle.dump(faces_data, f)

        with open('data/users.pkl', 'wb') as f:
            pickle.dump(users, f)

        st.success("Data wajah berhasil disimpan. Silakan latih model secara manual nanti.")

if __name__ == '__main__':
    main()
