import os
import pickle


def load_data(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data
    else:
        print(f"File {file_path} tidak ditemukan.")
        return None

def main():
    # Lokasi file .pkl
    faces_data_path = 'data/faces_data.pkl'
    users_data_path = 'data/users.pkl'

    # Muat data wajah
    faces_data = load_data(faces_data_path)
    if faces_data is not None:
        print("Isi faces_data.pkl:")
        print(faces_data)  # atau gunakan print(faces_data[:5]) untuk melihat beberapa data awal

    # Muat data pengguna
    users_data = load_data(users_data_path)
    if users_data is not None:
        print("\nIsi users.pkl:")
        for user in users_data:
            print(user)

if __name__ == "__main__":
    main()