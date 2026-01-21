"""Mathematische Hilfsfunktionen für Similarity-Berechnungen."""

from __future__ import annotations

from typing import Any

import numpy as np


def cosine_similarity(vektor_a: np.ndarray, vektor_b: np.ndarray) -> float:
    """
    Berechnet Kosinus-Ähnlichkeit cos(θ) = (a·b) / (||a||·||b||).

    Args:
        vektor_a (np.ndarray): Erster Vektor
        vektor_b (np.ndarray): Zweiter Vektor

    Returns:
        float: Similarity [-1, 1], bei Nullvektor 0.0
    """
    norm_a = float(np.linalg.norm(vektor_a))
    norm_b = float(np.linalg.norm(vektor_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(np.dot(vektor_a, vektor_b) / (norm_a * norm_b))


def relu(eingabe: np.ndarray) -> np.ndarray:
    """
    ReLU Aktivierungsfunktion: max(0, x).

    Args:
        eingabe (np.ndarray): Input-Array

    Returns:
        np.ndarray: max(0, x) elementweise
    """
    return np.maximum(0.0, eingabe)


def berechne_gewichtete_gesamt_similarity(
    similarity_pro_kategorie: dict[str, float],
    gewichtung_pro_kategorie: dict[str, float],
) -> float:
    """
    Gewichtetes Mittel: sum(w_i * sim_i) / sum(w_i).

    Args:
        similarity_pro_kategorie (dict): Kategorie -> Similarity
        gewichtung_pro_kategorie (dict): Kategorie -> Gewicht

    Returns:
        float: Gewichtetes Mittel, 0.0 bei sum(weights)=0
    """
    gewichtete_summe = 0.0
    gewicht_summe = 0.0

    for kategorie, gewicht in gewichtung_pro_kategorie.items():
        if gewicht <= 0.0:
            continue

        similarity = similarity_pro_kategorie.get(kategorie, 0.0)
        gewichtete_summe += gewicht * similarity
        gewicht_summe += gewicht

    return (gewichtete_summe / gewicht_summe) if gewicht_summe > 0 else 0.0


def normalisiere_embeddings_struktur(
    embeddings: Any,
    rotor_ids: list[str],
) -> dict[str, dict[str, np.ndarray]]:
    """
    Normalisiert Embedding-Strukturen auf einheitliches Format.

    Robust gegen 2 mögliche Strukturen:
    (A) embeddings[rotor_id][kategorie] = vector
    (B) embeddings[kategorie][rotor_id] = vector

    Args:
        embeddings (Any): Embeddings in beliebigem Format
        rotor_ids (list[str]): Liste aller Rotor-IDs

    Returns:
        dict: normalized[rotor_id][kategorie] = np.ndarray
    """
    if not embeddings:
        return {}

    if not isinstance(embeddings, dict):
        return {}

    erster_key = next(iter(embeddings.keys()))
    erster_wert = embeddings[erster_key]

    # Fall (A): embeddings[rotor_id] = {kat: vec}
    if isinstance(erster_wert, dict) and erster_key in rotor_ids:
        ausgabe: dict[str, dict[str, np.ndarray]] = {}
        for rotor_id in rotor_ids:
            kategorie_map = embeddings.get(rotor_id, {})
            ausgabe[rotor_id] = {
                kat: np.asarray(vec, dtype=float) for kat, vec in kategorie_map.items()
            }
        return ausgabe

    # Fall (B): embeddings[kat] = {rotor_id: vec}
    ausgabe_b: dict[str, dict[str, np.ndarray]] = {rid: {} for rid in rotor_ids}
    for kategorie, rotor_map in embeddings.items():
        if not isinstance(rotor_map, dict):
            continue
        for rotor_id, vektor in rotor_map.items():
            if rotor_id in ausgabe_b:
                ausgabe_b[rotor_id][kategorie] = np.asarray(vektor, dtype=float)
    return ausgabe_b
