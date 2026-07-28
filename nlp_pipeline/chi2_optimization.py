"""
==================================================================
Fase 4: Skenario 2 - Sesudah Optimasi dengan Feature Selection (Chi-Square)
Model: Multinomial Naive Bayes + SelectKBest (chi2)
==================================================================
"""

from typing import Any, Optional, Tuple, Dict, List
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_extraction.text import TfidfVectorizer


def select_features_chi2(
    X_train_tfidf: Any,
    y_train: Any,
    X_test_tfidf: Any,
    k: int = 1000,
) -> Tuple[Any, Any, Optional[SelectKBest], Optional[str]]:
    """
    Select top-k features using Chi-Square test.

    Parameters
    ----------
    X_train_tfidf : array-like or sparse matrix
        TF-IDF transformed training data.
    y_train : array-like
        Training labels.
    X_test_tfidf : array-like or sparse matrix
        TF-IDF transformed test data.
    k : int, default=1000
        Number of top features to select.

    Returns
    -------
    X_train_selected : array-like or sparse matrix
        Training data after feature selection.
    X_test_selected : array-like or sparse matrix
        Test data after feature selection.
    selector : SelectKBest or None
        Fitted selector (None if no features available or selection failed).
    info : str or None
        Informational/warning message.
    """
    # Determine number of features
    try:
        n_features_available = X_train_tfidf.shape[1]
    except AttributeError:
        # If it's not a sparse matrix or array with shape, try to get shape differently
        # For safety, we assume it's a matrix-like object with shape attribute
        n_features_available = 0

    # If no features at all, return original data and None selector
    if n_features_available == 0:
        info = (
            "Peringatan: Tidak ada fitur TF-IDF yang dihasilkan setelah preprocessing/TF-IDF. "
            "Feature selection dengan Chi-Square tidak dapat dilakukan."
        )
        return X_train_tfidf, X_test_tfidf, None, info

    # Determine actual k to use, ensuring at least one feature is selected
    k_actual = min(k, n_features_available)
    if k_actual == 0:
        # Force at least one feature to avoid empty model
        k_actual = 1
        info = (
            f"Peringatan: k={k} diminta tetapi hasilnya akan memilih 0 fitur. "
            f"Menggunakan k=1 fitur terbaik untuk menghindari model tanpa fitur."
        )
    else:
        info = None

    selector = SelectKBest(score_func=chi2, k=k_actual)
    X_train_selected = selector.fit_transform(X_train_tfidf, y_train)
    X_test_selected = selector.transform(X_test_tfidf)

    # If selection resulted in zero features, fall back to using all features
    if hasattr(X_train_selected, 'shape') and X_train_selected.shape[1] == 0:
        info = (
            f"Peringatan: Seleksi Chi-Square dengan k={k} menghasilkan 0 fitur. "
            f"Menggunakan semua {n_features_available} fitur sebagai fallback."
        )
        return X_train_tfidf, X_test_tfidf, None, info

    # Provide informative messages when needed
    if info is None:
        if k_actual < k:
            info = (
                f"k={k} diminta, tapi hanya {n_features_available} fitur tersedia "
                f"-> menggunakan k={k_actual}."
            )
        elif k_actual == n_features_available and k_actual < k:
            info = (
                f"Peringatan: k={k} diminta, tapi hanya {n_features_available} fitur tersedia. "
                f"Menggunakan semua {n_features_available} fitur (tidak ada pemilihan fitur yang dilakukan). "
                f"Untuk melakukan pemilihan fitur yang berarti, gunakan nilai k yang lebih kecil dari {n_features_available}."
            )

    return X_train_selected, X_test_selected, selector, info


def get_top_chi2_features(
    selector: Optional[SelectKBest],
    vectorizer: TfidfVectorizer,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Retrieve top features with their Chi‑square scores.

    Parameters
    ----------
    selector : SelectKBest or None
        Fitted selector from `select_features_chi2`.
    vectorizer : TfidfVectorizer
        The vectorizer used to produce the TF‑IDF matrix.
    top_n : int, default=20
        Number of top features to return.

    Returns
    -------
    pandas.DataFrame
        DataFrame with columns ['token', 'chi2_score'].
    """
    if selector is None:
        # No features selected – return empty DataFrame
        return pd.DataFrame(columns=["token", "chi2_score"])

    feature_names = vectorizer.get_feature_names_out()
    scores = selector.scores_
    support_mask = selector.get_support()

    df_scores = pd.DataFrame({
        "token": feature_names,
        "chi2_score": scores,
        "terpilih": support_mask,
    })

    top_features = (
        df_scores[df_scores["terpilih"]]
        .sort_values("chi2_score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    top_features.index += 1
    return top_features[["token", "chi2_score"]]


def buat_tabel_komparasi(
    n_fitur_sebelum: int,
    n_fitur_sesudah: int,
    hasil_sebelum: Dict[str, Any],
    hasil_sesudah: Dict[str, Any],
) -> pd.DataFrame:
    """
    Build a comparison DataFrame between baseline and chi‑square results.
    """
    data: Dict[str, List[float | int]] = {
        "Skenario 1 (Sebelum Optimasi)": [
            n_fitur_sebelum,
            hasil_sebelum["accuracy"],
            hasil_sebelum["precision"],
            hasil_sebelum["recall"],
            hasil_sebelum["f1_macro"],
            hasil_sebelum["f1_weighted"],
        ],
        "Skenario 2 (Sesudah Chi-Square)": [
            n_fitur_sesudah,
            hasil_sesudah["accuracy"],
            hasil_sesudah["precision"],
            hasil_sesudah["recall"],
            hasil_sesudah["f1_macro"],
            hasil_sesudah["f1_weighted"],
        ],
    }
    index = [
        "Jumlah Fitur",
        "Akurasi",
        "Presisi (Weighted)",
        "Recall (Weighted)",
        "F1-Score Macro",
        "F1-Score Weighted",
    ]

    df_komparasi = pd.DataFrame(data, index=index)
    df_komparasi["Selisih (Sesudah - Sebelum)"] = (
        df_komparasi["Skenario 2 (Sesudah Chi-Square)"]
        - df_komparasi["Skenario 1 (Sebelum Optimasi)"]
    )
    return df_komparasi.round(4)


def buat_analisis_teks(
    n_fitur_sebelum: int,
    n_fitur_sesudah: int,
    hasil_sebelum: Dict[str, Any],
    hasil_sesudah: Dict[str, Any],
) -> str:
    """Generate an automatic interpretation text (used for UI display)."""
    reduksi_persen = (1 - n_fitur_sesudah / n_fitur_sebelum) * 100
    delta_akurasi = hasil_sesudah["accuracy"] - hasil_sebelum["accuracy"]
    delta_f1_macro = hasil_sesudah["f1_macro"] - hasil_sebelum["f1_macro"]

    ambang = 0.001
    if delta_akurasi > ambang:
        teks_akurasi = f"meningkat sebesar {delta_akurasi:.4f}"
    elif delta_akurasi < -ambang:
        teks_akurasi = f"menurun sebesar {abs(delta_akurasi):.4f}"
    else:
        teks_akurasi = "relatif tidak berubah"

    if delta_f1_macro > ambang:
        teks_f1 = f"naik sebesar {delta_f1_macro:.4f}"
    elif delta_f1_macro < -ambang:
        teks_f1 = f"turun sebesar {abs(delta_f1_macro):.4f}"
    else:
        teks_f1 = "relatif stabil"

    paragraf1 = (
        f"Jumlah fiturusan dari **{n_fitur_sebelum}** menjadi **{n_fitur_sesudah}** "
        f"(reduksi dimensi sebesar **{reduksi_persen:.1f}%**) setelah seleksi Chi-Square."
    )
    paragraf2 = f"Akurasi model {teks_akurasi} dibanding baseline, dan F1-Score Macro {teks_f1}."

    if delta_akurasi >= -ambang and delta_f1_macro >= -ambang:
        kesimpulan = (
            "**Kesimpulan:** Seleksi fitur dengan Chi-Square berhasil memangkas dimensi ruang "
            "fitur TF-IDF secara signifikan tanpa mengorbankan performa model Naive Bayes -- "
            "bahkan pada beberapa kasus performanya sedikit lebih baik atau setara. Ini masuk akal "
            "karena TF-IDF pada teks ulasan pendek cenderung menghasilkan banyak token yang jarang "
            "muncul atau kurang relevan terhadap label sentimen (noise). Chi-Square membantu "
            "menyaring kata-kata yang secara statistik paling berasosiasi dengan kelas "
            "Positif/Netral/Negatif, sehingga model menjadi lebih ringkas dan komputasinya lebih "
            "efisien tanpa kehilangan daya prediksi."
        )
    else:
        kesimpulan = (
            "**Kesimpulan:** Meskipun dimensi fitur berhasil direduksi drastis, performa model "
            "justru sedikit menurun dibanding baseline. Ini bisa mengindikasikan nilai k yang "
            "dipilih terlalu kecil, sehingga sebagian token yang sebenarnya informatif ikut "
            "terbuang. Coba beberapa nilai k lain (misalnya 500, 2000, 5000) dan pilih titik yang "
            "memberi keseimbangan terbaik antara efisiensi komputasi dan performa model."
        )

    return f"{paragraf1}\n\n{paragraf2}\n\n{kesimpulan}"