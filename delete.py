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

def save_data(file_path, data):
    with open(file_path, 'wb') as file:
        pickle.dump(data, file)
    print(f"Data berhasil disimpan ke {file_path}")

def delete_face_data_by_user_id(faces_data, users_data, user_id_to_delete):
    # Menghapus wajah berdasarkan user_id yang sesuai dalam users_data
    updated_faces_data = []
    updated_users_data = []

    # Periksa apakah 'user_id' ada pada dictionary user
    for face, user in zip(faces_data, users_data):
        if user['user_id'] != user_id_to_delete:  # Menggunakan 'user_id' sebagai key
            updated_faces_data.append(face)
            updated_users_data.append(user)

    print(f"Data wajah dan pengguna dengan user_id {user_id_to_delete} berhasil dihapus.")
    return updated_faces_data, updated_users_data

def delete_user_data_by_username(users_data, username_to_delete):
    # Hapus pengguna berdasarkan username
    updated_users_data = [user for user in users_data if user['name'] != username_to_delete]  # Perbaiki key 'name='
    return updated_users_data

def main():
    # Lokasi file .pkl
    faces_data_path = 'data/faces_data.pkl'
    users_data_path = 'data/users.pkl'

    # Muat data wajah
    faces_data = load_data(faces_data_path)
    users_data = load_data(users_data_path)

    if faces_data is not None and users_data is not None:
        print("Isi faces_data.pkl dan users.pkl sebelum penghapusan:")
        print(f"Faces data: {len(faces_data)} item(s)")
        print(f"Users data: {len(users_data)} item(s)")

        # Misalkan kita ingin menghapus data berdasarkan user_id '5312422020'
        user_id_to_delete = '5312422020'  # Ganti dengan user_id yang ingin dihapus
        faces_data, users_data = delete_face_data_by_user_id(faces_data, users_data, user_id_to_delete)

        # Simpan kembali data yang telah diubah
        save_data(faces_data_path, faces_data)
        save_data(users_data_path, users_data)

        # Misalkan kita ingin menghapus pengguna dengan username 'Briska'
        username_to_delete = 'Briska'  # Ganti dengan username yang ingin dihapus
        users_data = delete_user_data_by_username(users_data, username_to_delete)

        # Simpan kembali data pengguna yang telah diubah
        save_data(users_data_path, users_data)

if __name__ == "__main__":
    main()