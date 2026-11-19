# TA-06: Prediksi Lagu Hit Spotify dengan Random Forest

## Isi folder
- `TA06_Random_Forest_Spotify_Hit.ipynb`: notebook lengkap dengan output (preprocessing, Skenario A dan B, RandomizedSearchCV, feature importance, evaluasi, simpan model)
- `app.py`, `templates/index.html`, `requirements.txt`, `Procfile`: aplikasi web **Flask**
- `model_hit_spotify.joblib` (72 MB): dibuat otomatis oleh notebook bagian 6 (Run All), harus berada di folder yang sama dengan `app.py`
- `grafik_*.png`, `hasil_skenario_*.csv`, `screenshot_web.png`: hasil eksperimen dan tampilan web
- `NASKAH_VIDEO_DEMO.md`: naskah video 3-5 menit

## Menjalankan ulang
1. Taruh `dataset.csv` (Kaggle: maharshipandya/spotify-tracks-dataset) di folder ini, jalankan notebook (Run All, sekitar 15 menit).
2. Web lokal: `pip install -r requirements.txt` lalu `python app.py`, buka http://localhost:5000
3. Deploy (contoh Render / Railway / PythonAnywhere): unggah folder ke GitHub (model 72 MB, di bawah batas 100 MB), lalu atur perintah start `gunicorn app:app --workers 1 --timeout 120` (sudah ada di `Procfile`). Pada Python versi berbeda, sesuaikan versi `scikit-learn` di `requirements.txt` dengan yang dipakai saat melatih model.

## Endpoint
- `GET /` halaman web
- `POST /predict` body JSON berisi `genre` dan 14 fitur audio, mengembalikan `probability`, `threshold`, `is_hit`
