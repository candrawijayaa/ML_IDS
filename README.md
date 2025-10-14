# ML_IDS

## Web Aplikasi (Flask)

Instruksi singkat untuk menjalankan antarmuka web berbasis Flask:

1. Buat dan aktifkan lingkungan virtual (opsional tapi disarankan):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
2. Pasang dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Jalankan server:
   ```bash
   python -m webapp.app
   ```/\
   - Untuk server produksi (contoh menggunakan gunicorn):
     ```bash
     gunicorn -w 1 -b 127.0.0.1:5000 webapp.app:app
     ```
4. Buka `http://127.0.0.1:5000` di peramban untuk mengunggah berkas PCAP (fitur akan diekstrak secara otomatis) atau gunakan endpoint `POST /api/predict` untuk prediksi berbasis JSON.
   - Jika port 5000 sedang terpakai, jalankan dengan `PORT=5050 python -m webapp.app` (bebas memilih port lain).

Model yang digunakan secara default berada di `models/model_random_forest.joblib`. Jika ingin mengganti model, ubah path pada pemanggilan `create_app`.

Catatan: proses ekstraksi PCAP membutuhkan dependensi `dpkt` (terpasang melalui `requirements.txt`).

Catatan: model dilatih menggunakan scikit-learn 1.7.x, sementara lingkungan runtime menggunakan 1.6.1 karena versi 1.7.x belum tersedia di arsip pip untuk platform ini. Saat memuat model, akan muncul peringatan `InconsistentVersionWarning` yang dapat diabaikan selama inferensi berjalan normal.
