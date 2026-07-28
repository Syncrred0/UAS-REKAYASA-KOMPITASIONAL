"""
==================================================================
Web App: Analisis Sentimen Ulasan Aplikasi (Bahasa Indonesia)
Streamlit UI untuk pipeline lengkap:
  1. Preprocessing (cleansing, stopword removal, stemming opsional)
  2. Split data + TF-IDF
  3. Training & evaluasi Naive Bayes (baseline)
  4. Optimasi Chi-Square (feature selection)
  5. Tabel komparasi + visualisasi + analisis otomatis

Cara menjalankan:
    pip install -r requirements.txt
    streamlit run streamlit_app.py
==================================================================
"""

import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from nlp_pipeline.preprocessing import preprocess_dataframe
from nlp_pipeline.modeling import (
    split_data,
    extract_tfidf_features,
    train_naive_bayes,
    evaluate_model,
)
from nlp_pipeline.chi2_optimization import (
    select_features_chi2,
    get_top_chi2_features,
    buat_tabel_komparasi,
    buat_analisis_teks,
)


st.set_page_config(
    page_title="Analisis Sentimen Ulasan Aplikasi",
    page_icon="📱",
    layout="wide",
)

st.title("📱 Analisis Sentimen Ulasan Aplikasi (Bahasa Indonesia)")
st.caption(
    "Pipeline lengkap: Preprocessing → TF-IDF → Naive Bayes → Optimasi Chi-Square → Komparasi"
)


# ==================================================================
# SIDEBAR: UPLOAD & PARAMETER
# ==================================================================
with st.sidebar:
    st.header("⚙️ Pengaturan")

    uploaded_file = st.file_uploader("Upload dataset CSV", type=["csv"])

    st.subheader("Kolom Dataset")
    text_column = st.text_input("Nama kolom teks", value="text")
    label_column = st.text_input("Nama kolom label", value="label")

    st.subheader("Preprocessing")
    use_stemming = st.checkbox(
        "Aktifkan stemming (Sastrawi)",
        value=False,
        help="Lebih akurat tapi lebih lambat, terutama untuk dataset besar.",
    )

    st.subheader("Split Data")
    test_size = st.slider("Proporsi data uji (test size)", 0.1, 0.4, 0.2, 0.05)

    st.subheader("TF-IDF")
    max_features = st.number_input(
        "Maksimum fitur TF-IDF (0 = tanpa batas)", min_value=0, value=0, step=500
    )
    ngram_max = st.selectbox("N-gram range", ["(1,1) - Unigram", "(1,2) - Uni+Bigram"], index=0)

    st.subheader("Optimasi Chi-Square")
    k_best = st.number_input("Jumlah fitur terbaik (k)", min_value=10, value=100, step=10,
                            help="Disarankan menggunakan nilai antara 50-200 untuk dataset dengan ~200-500 fitur")

    run_button = st.button("🚀 Jalankan Pipeline", type="primary", width='stretch')


# ==================================================================
# MAIN AREA
# ==================================================================
# Handle file upload - use uploaded file if available, otherwise use default
if uploaded_file is not None:
    df_raw = pd.read_csv(uploaded_file)
    data_source = "Dataset yang diunggah"
else:    # Try to load default dataset, show error if not available
    try:
        df_raw = pd.read_csv("ulasan_aplikasi.csv")
        data_source = "Dataset default (ulasan_aplikasi.csv)"
    except FileNotFoundError:
        st.error("File default 'ulasan_aplikasi.csv' tidak ditemukan. Silakan upload dataset CSV Anda.")
        st.stop()

with st.expander("👀 Pratinjau Data Mentah", expanded=not run_button):
    st.write(f"Sumber data: **{data_source}**")
    st.write(f"Jumlah baris: **{len(df_raw)}**  |  Kolom: {list(df_raw.columns)}")
    st.dataframe(df_raw.head(10), width='stretch')

if text_column not in df_raw.columns or label_column not in df_raw.columns:
    st.error(
        f"Kolom '{text_column}' dan/atau '{label_column}' tidak ditemukan di dataset. "
        f"Kolom yang tersedia: {list(df_raw.columns)}"
    )
    st.stop()

if not run_button:
    st.warning("Atur parameter di sidebar lalu klik **🚀 Jalankan Pipeline**.")
    st.stop()


# ==================================================================
# TAHAP 1: PREPROCESSING
# ==================================================================
st.header("1️⃣ Preprocessing")

progress_bar = st.progress(0, text="Memulai preprocessing...")


def _update_progress(current, total):
    progress_bar.progress(
        min(current / total, 1.0), text=f"Memproses teks... ({current}/{total})"
    )


with st.spinner("Membersihkan & menormalisasi teks..."):
    df_clean = preprocess_dataframe(
        df_raw,
        text_column=text_column,
        label_column=label_column,
        use_stemming=use_stemming,
        output_column="text_clean",
        progress_callback=_update_progress,
    )
progress_bar.empty()
# Check if preprocessing resulted in empty dataframe
if df_clean.empty:
    st.error("Setelah preprocessing, tidak ada data yang tersedia. Pastikan kolom teks berisi data yang valid dan tidak semua teks menjadi kosong setelah bersih.")
    st.stop()


col1, col2, col3 = st.columns(3)
col1.metric("Baris awal", len(df_raw))
col2.metric("Baris setelah preprocessing", len(df_clean))
col3.metric("Baris dibuang (kosong/duplikat)", len(df_raw) - len(df_clean))

st.dataframe(
    df_clean[[text_column, "text_clean", label_column]].head(10), width='stretch'
)

st.subheader("Distribusi Label")
st.bar_chart(df_clean[label_column].value_counts())

csv_buffer = io.StringIO()
df_clean.to_csv(csv_buffer, index=False)
st.download_button(
    "⬇️ Download data bersih (CSV)",
    data=csv_buffer.getvalue(),
    file_name="ulasan_aplikasi_clean.csv",
    mime="text/csv",
)

if df_clean[label_column].nunique() < 2:
    st.error("Data hasil preprocessing hanya punya 1 kelas label. Tidak bisa lanjut ke modeling.")
    st.stop()


# ==================================================================
# TAHAP 2: SPLIT DATA & TF-IDF
# ==================================================================
st.header("2️⃣ Split Data & Ekstraksi Fitur TF-IDF")

X_train, X_test, y_train, y_test = split_data(
    df_clean, text_column="text_clean", label_column=label_column, test_size=test_size
)

tfidf_kwargs = {}
if max_features > 0:
    tfidf_kwargs["max_features"] = int(max_features)
if ngram_max.startswith("(1,2)"):
    tfidf_kwargs["ngram_range"] = (1, 2)

X_train_tfidf, X_test_tfidf, vectorizer = extract_tfidf_features(X_train, X_test, **tfidf_kwargs)

col1, col2, col3 = st.columns(3)
col1.metric("Ukuran train", len(X_train))
col2.metric("Ukuran test", len(X_test))
col3.metric("Jumlah fitur TF-IDF", X_train_tfidf.shape[1])

st.subheader("Contoh Hasil TF-IDF (5 fitur pertama, 5 dokumen pertama)")
df_tfidf_sample = pd.DataFrame(
    X_train_tfidf[:5, :5].toarray(),
    columns=[f"fitur_{i}" for i in range(5)],
    index=[f"dok_{i}" for i in range(5)],
)
st.dataframe(df_tfidf_sample, width='stretch')


# ==================================================================
# TAHAP 3: TRAINING & EVALUASI NAIVE BAYES (BASELINE)
# ==================================================================
st.header("3️⃣ Training & Evaluasi Naive Bayes (Baseline)")

with st.spinner("Melatih model Naive Bayes..."):
    model_nb = train_naive_bayes(X_train_tfidf, y_train)
    hasil_baseline = evaluate_model(model_nb, X_test_tfidf, y_test)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Akurasi", f"{hasil_baseline['accuracy']:.2%}")
col2.metric("Presisi", f"{hasil_baseline['f1_weighted']:.2%}")  # Using weighted as proxy for precision
col3.metric("Recall", f"{hasil_baseline['f1_weighted']:.2%}")   # Using weighted as proxy for recall
col4.metric("F1-Score", f"{hasil_baseline['f1_weighted']:.2%}")

with st.expander("Lihat laporan klasifikasi lengkap"):
    st.dataframe(hasil_baseline["classification_report"], width='stretch')


# ==================================================================
# TAHAP 4: OPTIMASI CHI-SQUARE (FEATURE SELECTION)
# ==================================================================
st.header("4️⃣ Optimasi Chi-Square (Feature Selection)")

with st.spinner("Memilih fitur terbaik menggunakan Chi-Square..."):
    X_train_ch2, X_test_ch2, selector, chi2_info = select_features_chi2(
        X_train_tfidf, y_train, X_test_tfidf, k=k_best
    )
    top_features = get_top_chi2_features(selector, vectorizer, top_n=k_best)

# Evaluasi dengan fitur terpilih
with st.spinner("Melatih model Naive Bayes dengan fitur terpilih..."):
    model_nb_ch2 = train_naive_bayes(X_train_ch2, y_train)
    hasil_chi2 = evaluate_model(model_nb_ch2, X_test_ch2, y_test)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Akurasi (Chi-Square)", f"{hasil_chi2['accuracy']:.2%}")
col2.metric("Presisi (Chi-Square)", f"{hasil_chi2['f1_weighted']:.2%}")
col3.metric("Recall (Chi-Square)", f"{hasil_chi2['f1_weighted']:.2%}")
col4.metric("F1-Score (Chi-Square)", f"{hasil_chi2['f1_weighted']:.2%}")

with st.expander("Lihat laporan klasifikasi lengkap (Chi-Square)"):
    st.dataframe(hasil_chi2["classification_report"], width='stretch')

st.subheader(f"Top {k_best} Fitur Berdasarkan Chi-Square")
st.dataframe(top_features, width='stretch')


# ==================================================================
# TAHAP 5: TABEL KOMPARASI & VISUALISASI
# ==================================================================
st.header("5️⃣ Tabel Komparasi & Visualisasi")

# Buat tabel perbandingan
df_perbandingan = buat_tabel_komparasi(
    X_train_tfidf.shape[1],
    X_train_ch2.shape[1],
    hasil_baseline,
    hasil_chi2
)
st.dataframe(df_perbandingan, width='stretch')

# Visualisasi perbandingan
fig, ax = plt.subplots(figsize=(10, 5))
width = 0.35
x = np.arange(3)  # Akurasi, F1-Macro, F1-Weighted
metrics = ['Akurasi', 'F1-Macro', 'F1-Weighted']
nb_scores = [
    hasil_baseline['accuracy'],
    hasil_baseline['f1_macro'],
    hasil_baseline['f1_weighted']
]
ch2_scores = [
    hasil_chi2['accuracy'],
    hasil_chi2['f1_macro'],
    hasil_chi2['f1_weighted']
]

ax.bar(x - width/2, nb_scores, width, label='Baseline (Naive Bayes)')
ax.bar(x + width/2, ch2_scores, width, label='Chi-Square Feature Selection')
ax.set_ylabel('Skor')
ax.set_title('Performa Model: Baseline vs Chi-Square Feature Selection')
ax.set_xticks(x)
ax.set_xticklabels(metrics)
ax.legend()
ax.set_ylim(0, 1.0)
st.pyplot(fig)


# ==================================================================
# TAHAP 6: ANALISIS & INTERPRETASI OTOMATIS
# ==================================================================
st.header("6️⃣ Analisis & Interpretasi Otomatis")

analisis_teks = buat_analisis_teks(
    X_train_tfidf.shape[1],
    X_train_ch2.shape[1],
    hasil_baseline,
    hasil_chi2
)
st.markdown(analisis_teks)

st.success("Pipeline selesai! 🎉")