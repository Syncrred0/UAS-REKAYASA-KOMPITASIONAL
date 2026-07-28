"""
==================================================================
Fase 1: Preprocessing Data
Proyek: Analisis Sentimen Ulasan Aplikasi (Bahasa Indonesia)
==================================================================
"""

import re
import pandas as pd
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


# ==================================================================
# 1. INISIALISASI SASTRAWI (stopword remover & stemmer)
# ==================================================================
_stopword_factory = StopWordRemoverFactory()
_default_stopwords = set(_stopword_factory.get_stop_words())

custom_stopwords = {
    "nya", "yg", "dg", "dgn", "rt", "d", "klo", "kalo", "amp",
    "biar", "bikin", "bilang", "gak", "ga", "krn", "nih", "sih",
    "si", "tau", "tdk", "tuh", "utk", "ya", "jd", "jgn", "sdh",
    "aja", "n", "t", "hehe", "u", "loh", "deh", "dong", "kok",
}
STOPWORDS = _default_stopwords.union(custom_stopwords)

_stemmer_factory = StemmerFactory()
_stemmer = _stemmer_factory.create_stemmer()


# ==================================================================
# 2. FUNGSI-FUNGSI PREPROCESSING (per tahap)
# ==================================================================

def case_folding(text: str) -> str:
    return text.lower()


def remove_emoji(text: str) -> str:
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"
        "\U0001F300-\U0001F5FF"
        "\U0001F680-\U0001F6FF"
        "\U0001F1E0-\U0001F1FF"
        "\U00002700-\U000027BF"
        "\U0001F900-\U0001F9FF"
        "\U00002600-\U000026FF"
        "\U0001FA70-\U0001FAFF"
        "]+",
        flags=re.UNICODE,
    )
    return emoji_pattern.sub(r" ", text)


def cleansing(text: str) -> str:
    text = remove_emoji(text)
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+|#\w+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"_", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text: str) -> str:
    tokens = text.split()
    tokens_bersih = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens_bersih)


def stemming(text: str) -> str:
    return _stemmer.stem(text)


def preprocess_text(text: str, use_stemming: bool = False) -> str:
    if not isinstance(text, str) or text.strip() == "":
        return ""

    text = case_folding(text)
    text = cleansing(text)
    text = remove_stopwords(text)

    if use_stemming:
        text = stemming(text)

    return text


# ==================================================================
# 3. FUNGSI UNTUK MEMPROSES SELURUH DATAFRAME
# ==================================================================

def preprocess_dataframe(
    df: pd.DataFrame,
    text_column: str = "text",
    label_column: str = "label",
    use_stemming: bool = False,
    output_column: str = "text_clean",
    progress_callback=None,
    return_stats: bool = False,
):
    """
    Menjalankan preprocessing pada seluruh kolom teks dalam DataFrame.

    progress_callback : callable(current, total), optional
        Dipanggil setiap baris selesai diproses -> dipakai Streamlit
        untuk menampilkan progress bar.
    return_stats : bool, optional
        Jika True, mengembalikan tuple (df, stats) dengan `stats` berisi
        rincian jumlah baris yang hilang di tiap tahap (kosong vs duplikat)
        -> berguna untuk mendiagnosis kenapa dataset menyusut drastis.
    """
    df = df.copy()
    stats = {"baris_awal": len(df)}

    # Pastikan kolom yang diperlukan ada
    if text_column not in df.columns:
        raise ValueError(f"Text column '{text_column}' tidak ditemukan dalam dataframe")
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' tidak ditemukan dalam dataframe")

    # Hapus baris dengan nilai missing pada text atau label
    df = df.dropna(subset=[text_column, label_column])

    # Hapus baris dengan teks kosong (setelah strip)
    df = df[df[text_column].astype(str).str.strip() != ""]
    # Hapus baris dengan label kosong (setelah strip) - menangani label berbasis string
    df = df[df[label_column].astype(str).str.strip() != ""]

    df = df.reset_index(drop=True)
    stats["setelah_hapus_kosong"] = len(df)

    total = len(df)
    hasil = []
    for i, teks in enumerate(df[text_column].tolist()):
        hasil.append(preprocess_text(teks, use_stemming=use_stemming))
        if progress_callback is not None and (i % 25 == 0 or i == total - 1):
            progress_callback(i + 1, total)

    df[output_column] = hasil

    # Pastikan kita hanya menyimpan baris yang memiliki teks yang diproses yang tidak kosong
    # dan mempertahankan kolom label
    df = df[df[output_column].str.strip() != ""]
    stats["setelah_hapus_hasil_kosong"] = len(df)

    df = df.drop_duplicates(subset=[output_column]).reset_index(drop=True)
    stats["setelah_hapus_duplikat"] = len(df)
    stats["baris_duplikat_dibuang"] = (
        stats["setelah_hapus_hasil_kosong"] - stats["setelah_hapus_duplikat"]
    )

    if return_stats:
        return df, stats
    return df