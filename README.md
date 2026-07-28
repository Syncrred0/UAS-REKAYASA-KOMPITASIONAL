# 📱 Analisis Sentimen Ulasan Aplikasi — Web App (Streamlit)

Web app ini menggabungkan 4 skrip Python yang sudah ada menjadi satu pipeline
interaktif berbasis **Streamlit**:

| Skrip asli | Peran di web app |
|---|---|
| `app.py` | Fase 1 — Preprocessing (cleansing, stopword removal, stemming) |
| `modeling.py` | Fase 2 & 3 — Split data, TF-IDF, Naive Bayes, evaluasi |
| `optimasi.py` / `optimasi_chi2.py` | Fase 4 — Feature selection Chi-Square, komparasi, visualisasi |

## 📂 Struktur Folder

```
sentiment_webapp/
├── streamlit_app.py          # Entry point web app
├── requirements.txt
├── nlp_pipeline/
│   ├── __init__.py
│   ├── preprocessing.py      # dari app.py
│   ├── modeling.py           # dari modeling.py
│   └── chi2_optimization.py  # dari optimasi.py / optimasi_chi2.py
└── README.md
```

## 🚀 Cara Menjalankan

1. **Buat virtual environment (disarankan)**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Jalankan aplikasi**
   ```bash
   streamlit run streamlit_app.py
   ```

4. Browser akan otomatis terbuka di `http://localhost:8501`.

## 🖱️ Cara Pakai

1. Upload file CSV berisi ulasan di sidebar (minimal ada kolom teks & kolom label).
2. Sesuaikan nama kolom teks/label jika berbeda dari default (`text`, `label`).
3. Atur parameter di sidebar:
   - Aktifkan/matikan **stemming**
   - Proporsi **data uji** (test size)
   - Batas maksimum fitur TF-IDF & n-gram range
   - Jumlah fitur terbaik **k** untuk Chi-Square
4. Klik **🚀 Jalankan Pipeline**.
5. Hasil akan tampil berurutan:
   - Preprocessing (pratinjau teks bersih + distribusi label + download CSV)
   - Skenario 1: Baseline Naive Bayes (seluruh fitur TF-IDF)
   - Skenario 2: Naive Bayes + Chi-Square feature selection
   - Tabel komparasi & grafik batang
   - Analisis & interpretasi otomatis

## 📝 Format Dataset

CSV minimal punya 2 kolom, contoh:

| text | label |
|---|---|
| Aplikasinya bagus banget, gampang dipakai! | positif |
| Lemot dan sering force close | negatif |
| Biasa aja, standar | netral |

## ⚠️ Catatan

- Stemming (Sastrawi) cukup berat secara komputasi — untuk dataset besar
  (>5.000 baris), sebaiknya nonaktifkan dulu saat eksperimen awal.
- Instalasi pertama `Sastrawi` mungkin butuh waktu beberapa detik untuk
  mengunduh kamus kata dasar.
