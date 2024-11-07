import pickle

# Buka dan muat data dari file faces_data.pkl
file_path = 'data/faces_data.pkl'
with open(file_path, 'rb') as file:
    faces_data = pickle.load(file)

# Tampilkan data untuk melihat formatnya
print("Data Wajah yang Ada:")
print(faces_data)

# Tentukan user_id yang ingin dihapus
user_id_to_remove = '5312422020'

# Perbarui data wajah dengan menghapus entri sesuai user_id
faces_data = [data for data in faces_data if data['user_id'] != user_id_to_remove]

# Simpan ulang data wajah yang sudah diperbarui ke file faces_data.pkl
with open(file_path, 'wb') as file:
    pickle.dump(faces_data, file)

print("\nData wajah telah diperbarui, entri dengan user_id:", user_id_to_remove, "telah dihapus.")
