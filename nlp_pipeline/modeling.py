"""
==================================================================
Fase 2: Ekstraksi Fitur (TF-IDF) & Split Data
Fase 3: Modeling & Evaluasi - Skenario Sebelum Optimasi
Model: Multinomial Naive Bayes
==================================================================
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    classification_report,
    confusion_matrix,
)
import streamlit as st

RANDOM_STATE = 42


def split_data(
    df: pd.DataFrame,
    text_column: str = "text_clean",
    label_column: str = "label",
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
):
    X = df[text_column]
    y = df[label_column]

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y,
        )
    except ValueError as e:
        # If stratification fails due to insufficient class members, try without stratification
        st.warning(f"Stratified split failed: {e}. Using non-stratified split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
        )
    return X_train, X_test, y_train, y_test


def extract_tfidf_features(X_train, X_test, **tfidf_kwargs):
    vectorizer = TfidfVectorizer(**tfidf_kwargs)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    return X_train_tfidf, X_test_tfidf, vectorizer


def train_naive_bayes(X_train_feat, y_train):
    model = MultinomialNB()
    model.fit(X_train_feat, y_train)
    return model


def evaluate_model(model, X_test_feat, y_test, label_order=None):
    """
    Mengembalikan dict berisi semua metrik + classification report (dict)
    + confusion matrix (DataFrame), tanpa melakukan print apa pun -> agar
    fleksibel dipakai baik di skrip CLI maupun Streamlit.
    """
    y_pred = model.predict(X_test_feat)

    if label_order is None:
        label_order = sorted(y_test.unique())

    accuracy = accuracy_score(y_test, y_pred)
    precision_weighted = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    report_dict = classification_report(
        y_test, y_pred, labels=label_order, digits=4, output_dict=True, zero_division=0
    )
    report_df = pd.DataFrame(report_dict).transpose()

    cm = confusion_matrix(y_test, y_pred, labels=label_order)
    cm_df = pd.DataFrame(
        cm,
        index=[f"Aktual: {l}" for l in label_order],
        columns=[f"Prediksi: {l}" for l in label_order],
    )

    # Diagnostik ukuran data uji -> membantu menjelaskan skor 0%/100% yang
    # murni kebetulan ketika data uji (atau salah satu kelasnya) terlalu kecil.
    test_label_counts = y_test.value_counts()
    n_test = len(y_test)
    min_class_count = int(test_label_counts.min()) if n_test > 0 else 0
    reliability_warning = None
    if n_test < 10 or min_class_count < 3:
        reliability_warning = (
            f"Data uji hanya berisi {n_test} baris (kelas paling sedikit: "
            f"{min_class_count} baris). Dengan data sekecil ini, akurasi/F1-Score "
            f"bisa saja 0% atau 100% murni karena kebetulan, bukan cerminan performa "
            f"model yang sesungguhnya. Sebaiknya gunakan dataset yang lebih besar."
        )

    return {
        "y_pred": y_pred,
        "accuracy": accuracy,
        "precision": precision_weighted,
        "recall": recall_weighted,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "classification_report": report_df,
        "confusion_matrix": cm_df,
        "label_order": label_order,
        "n_test": n_test,
        "test_label_counts": test_label_counts,
        "reliability_warning": reliability_warning,
    }