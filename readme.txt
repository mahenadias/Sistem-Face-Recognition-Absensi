add_faces.py		: program untuk menambahkan data wajah baru tanpa training cnn
add_faces_streamlit.py	: program untuk menambahkan data wajah baru sekaligus training cnn
main.py			: program main semua sistem
read_pickle.py		: program untuk membaca atau mengedit file .pkl
show_attendance.py	: program untuk melihat data kehadiran dari output file JSON Temporary yang dihasilkan dari take_attendance_cnn.py
take_attendance_cnn.py	: program untuk mengambil daftar kehadiran berdasarkan wajah yang dikenali dan menyelesaikan task (berkedip)
train_cnn_model.py	: program untuk mentraining data wajah yang sudah ditambahkan (mentraining data file .pkl)

Note:
Jangan lupa tambahkan data wajah baru terlebih dahulu, dan jalankan training cnn nya untuk dapat menjalankan file program take_attendance_cnn.py