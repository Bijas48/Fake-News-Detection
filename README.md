# Detektor Berita Palsu

Aplikasi ini menggunakan model **Logistic Regression** untuk mendeteksi apakah sebuah berita adalah **asli** atau **palsu**. Dibangun menggunakan **Streamlit** untuk antarmuka pengguna.

## Fitur
- Input judul berita untuk diperiksa.
- Deteksi berita asli atau palsu menggunakan model yang telah dilatih.

## Persyaratan
Pastikan Anda memiliki:
- Python 3.8 atau lebih baru
- Virtual environment (opsional, tetapi disarankan)

## Instalasi
1. Clone repositori ini:
   ```bash
   git clone https://github.com/Bijas48/Fake-News-Detection.git
   cd testing_model-Bert
    ```
2. Buat dan aktifkan virtual environment (opsional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Untuk Linux/Mac
   venv\Scripts\activate  # Untuk Windows
   ```
3. Instal dependensi:
   ```bash
   pip install -r requirements.txt
   ``` 
4. Jalankan aplikasi Streamlit:
   ```bash
   streamlit run app.py
   ```
5. Buka browser dan akses `http://localhost:8501` untuk melihat aplikasi.


## Cara Kerja
1. Pengguna memasukkan judul berita ke dalam kolom input.
2. Setelah menekan tombol "Deteksi", aplikasi akan memproses input dan menggunakan model Logistic Regression untuk menentukan apakah berita tersebut asli atau palsu.
3. Hasil deteksi akan ditampilkan di layar.
4. Pengguna dapat mengulangi langkah 1-3 untuk memeriksa berita lainnya.



