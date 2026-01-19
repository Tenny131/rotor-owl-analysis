import pytest

from rotor_owl.utils.aufbereitung import normalize_param_name, safe_float


@pytest.mark.parametrize(
    "parameter_name_eingabe, parameter_name_erwartet",
    [
        ("P_WELLE_TIR_D001", "P_WELLE_TIR"),
        ("P_WELLE_TIR_2025-11-30_1", "P_WELLE_TIR"),
        ("P_AKTIV_LAENGE_XYZ", "P_AKTIV_LAENGE"),
        ("P_LUEFTER_D", "P_LUEFTER"),  # fallback: entfernt letztes _Segment
    ],
)
def test_normalize_param_name(parameter_name_eingabe: str, parameter_name_erwartet: str) -> None:
    assert normalize_param_name(parameter_name_eingabe) == parameter_name_erwartet


@pytest.mark.parametrize(
    "wert_eingabe, wert_erwartet",
    [
        ("12.5", 12.5),
        ("12,5", 12.5),
        ('"12.5"', 12.5),
        ("", None),
        ("NaN", None),
        (None, None),
        ("ABC", None),
    ],
)
def test_safe_float(wert_eingabe, wert_erwartet) -> None:
    assert safe_float(wert_eingabe) == wert_erwartet
