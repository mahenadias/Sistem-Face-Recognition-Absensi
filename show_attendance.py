import json
import os
from datetime import datetime, timedelta

import streamlit as st


def remove_expired_attendance(file_path):
    if not os.path.exists(file_path):
        return []

    with open(file_path, 'r') as f:
        attendance_data = json.load(f)

    one_hour_ago = datetime.now() - timedelta(hours=1)
    updated_data = [entry for entry in attendance_data if datetime.strptime(entry["Waktu Kehadiran"], "%Y-%m-%d_%H-%M-%S") > one_hour_ago]

    with open(file_path, 'w') as f:
        json.dump(updated_data, f)

    return updated_data

def main():
    st.title("Daftar Kehadiran Sementara")

    file_path = "data/attendance_temp.json"

    attendance_data = remove_expired_attendance(file_path)

    if attendance_data:
        st.json(attendance_data)
    else:
        st.warning("Belum ada data kehadiran yang tercatat dalam satu jam terakhir.")

if __name__ == "__main__":
    main()
