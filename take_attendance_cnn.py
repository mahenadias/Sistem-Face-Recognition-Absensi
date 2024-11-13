import json
import os
import pickle
import time
from datetime import datetime

import cv2
import dlib
import numpy as np
import streamlit as st
from imutils import face_utils
from keras.models import load_model
from PIL import Image


# Fungsi untuk memuat model dan data pengguna
def load_model_and_data():
    model = load_model('data/face_recognition_cnn.h5')
    with open('data/users.pkl', 'rb') as f:
        users = pickle.load(f)
    return model, users

def record_attendance_json(name, user_id, major, full_frame):
    file_path = 'data/attendance_temp.json'
    
    if not os.path.exists('data/attendance_images'):
        os.makedirs('data/attendance_images')

    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    image_filename = f"data/attendance_images/{name}_{current_time}.jpg"
    cv2.imwrite(image_filename, full_frame)

    new_entry = {
        "Nama": name,
        "NIM": user_id,
        "Prodi": major,
        "Waktu Kehadiran": current_time,
        "Gambar": image_filename
    }

    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            attendance_data = json.load(f)
    else:
        attendance_data = []

    attendance_data.append(new_entry)

    with open(file_path, 'w') as f:
        json.dump(attendance_data, f)

def predict_face(face, model, users, threshold=0.6, confidence_threshold=0):  
    face = cv2.resize(face, (64, 64))  
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = face.astype('float32') / 255.0
    face = np.expand_dims(face, axis=-1)
    face = np.expand_dims(face, axis=0)
    
    predictions = model.predict(face)
    max_pred = np.max(predictions)

    if max_pred >= threshold and max_pred >= confidence_threshold:
        predicted_class = np.argmax(predictions)
        predicted_name = users[predicted_class]['name']
        user_id = users[predicted_class]['user_id']
        major = users[predicted_class]['major']
        return predicted_name, user_id, major, max_pred
    else:
        return "Tidak Dikenali", None, None, max_pred

def calculate_ear(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def calibrate_ear_threshold(cap, predictor, face_cascade):
    ear_values = []
    for _ in range(30):  # Ambil 30 sampel
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            left_eye = shape[36:42]
            right_eye = shape[42:48]

            left_ear = calculate_ear(left_eye)
            right_ear = calculate_ear(right_eye)
            ear_values.append((left_ear + right_ear) / 2.0)

    average_ear = np.mean(ear_values)
    return average_ear * 0.8  # Gunakan 80% dari rata-rata EAR sebagai threshold

def is_eye_blinking(eye, threshold):
    ear = calculate_ear(eye)
    return ear < threshold

def take_attendance():
    st.title("Sistem Absensi Pengenalan Wajah")
    st.write("Aktifkan kamera untuk melakukan absensi.")

    # Tetapkan nilai default threshold tanpa slider
    threshold = 0.8
    confidence_threshold = 0.8

    model, users = load_model_and_data()

    cap = cv2.VideoCapture(1)
    frame_placeholder = st.empty()
    comment_placeholder = st.empty()

    last_recorded_time = 0
    predictor = dlib.shape_predictor("data/shape_predictor_68_face_landmarks.dat")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Kalibrasi threshold EAR
    EAR_THRESHOLD = calibrate_ear_threshold(cap, predictor, face_cascade)
    st.write(f"EAR Threshold Terkalibrasi: {EAR_THRESHOLD:.2f}")

    task = "Kedipkan mata"
    task_completed = False
    task_start_time = time.time()
    message_displayed_until = 0  # Waktu hingga kapan pesan ditampilkan
    consecutive_blinks = 0  # Hitungan kedipan berturut-turut
    CONSEC_FRAMES = 3  # Jumlah frame berturut-turut untuk mendeteksi kedipan
    blink_detected = False  # Menandai kedipan yang sudah terdeteksi

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Gagal membuka kamera")
            break

        current_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            comment_placeholder.write("Wajah tidak dikenali.")
        else:
            # Proses hanya wajah pertama yang terdeteksi
            x, y, w, h = faces[0]
            rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            left_eye = shape[36:42]
            right_eye = shape[42:48]

            name, user_id, major, confidence = predict_face(
                frame[y:y+h, x:x+w], model, users, threshold=threshold, confidence_threshold=confidence_threshold
            )

            if name != "Tidak Dikenali":
                cv2.putText(frame, f"Nama: {name}", (x, y-60), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"NIM: {user_id}", (x, y-40), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(frame, f"Prodi: {major}", (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (255, 255, 255), 1)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                if task == "Kedipkan mata" and (
                    is_eye_blinking(left_eye, EAR_THRESHOLD) or is_eye_blinking(right_eye, EAR_THRESHOLD)
                ):
                    consecutive_blinks += 1
                    if consecutive_blinks >= CONSEC_FRAMES and not blink_detected:
                        task_completed = True
                        blink_detected = True
                        consecutive_blinks = 0
                        comment_placeholder.write("Tugas selesai: Kedipkan mata.")
                else:
                    if blink_detected and not is_eye_blinking(left_eye, EAR_THRESHOLD) and not is_eye_blinking(right_eye, EAR_THRESHOLD):
                        blink_detected = False
                    consecutive_blinks = 0

                if task_completed and current_time - last_recorded_time >= 3:
                    record_attendance_json(name, user_id, major, frame)
                    comment_placeholder.write("Wajah dikenali dan tercatat oleh sistem.")
                    last_recorded_time = current_time
                    message_displayed_until = current_time + 5
                    task_completed = False
            else:
                cv2.putText(frame, "Wajah Tidak Dikenali", (x, y-20), cv2.FONT_HERSHEY_COMPLEX, 0.6, (0, 0, 255), 1)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                comment_placeholder.write("Wajah tidak dikenali.")

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(frame_rgb)
        frame_placeholder.image(img_pil, use_column_width=True)

        if current_time < message_displayed_until:
            comment_placeholder.write("Wajah dikenali dan tercatat oleh sistem.")

        if not task_completed and current_time - task_start_time > 3:
            comment_placeholder.write(f"Lakukan tindakan: {task}")
            task_start_time = current_time

        time.sleep(0.1)

    cap.release()

def main():
    take_attendance()

if __name__ == "__main__":
    main()