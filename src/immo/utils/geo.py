"""French geographic reference data and utilities.

Contains comprehensive mappings of regions, departments, and helper
functions for geographic lookups.
"""

from __future__ import annotations



# ---------------------------------------------------------------------------
# Regions -> department numbers (metropolitan France + DOM)
# ---------------------------------------------------------------------------

REGIONS: dict[str, list[int]] = {
    "Auvergne-Rhone-Alpes": [1, 3, 7, 15, 26, 38, 42, 43, 63, 69, 73, 74],
    "Bourgogne-Franche-Comte": [21, 25, 39, 58, 70, 71, 89, 90],
    "Bretagne": [22, 29, 35, 56],
    "Centre-Val de Loire": [18, 28, 36, 37, 41, 45],
    "Corse": [201, 202],  # 2A=201, 2B=202 (numeric convention)
    "Grand Est": [8, 10, 51, 52, 54, 55, 57, 67, 68, 88],
    "Hauts-de-France": [2, 59, 60, 62, 80],
    "Ile-de-France": [75, 77, 78, 91, 92, 93, 94, 95],
    "Normandie": [14, 27, 50, 61, 76],
    "Nouvelle-Aquitaine": [16, 17, 19, 23, 24, 33, 40, 47, 64, 79, 86, 87],
    "Occitanie": [9, 11, 12, 30, 31, 32, 34, 46, 48, 65, 66, 81, 82],
    "Pays de la Loire": [44, 49, 53, 72, 85],
    "Provence-Alpes-Cote d'Azur": [4, 5, 6, 13, 83, 84],
    # DOM-TOM
    "Guadeloupe": [971],
    "Martinique": [972],
    "Guyane": [973],
    "La Reunion": [974],
    "Mayotte": [976],
}


# ---------------------------------------------------------------------------
# Department number -> name
# ---------------------------------------------------------------------------

DEPARTMENTS: dict[int, str] = {
    1: "Ain", 2: "Aisne", 3: "Allier", 4: "Alpes-de-Haute-Provence",
    5: "Hautes-Alpes", 6: "Alpes-Maritimes", 7: "Ardeche", 8: "Ardennes",
    9: "Ariege", 10: "Aube", 11: "Aude", 12: "Aveyron",
    13: "Bouches-du-Rhone", 14: "Calvados", 15: "Cantal", 16: "Charente",
    17: "Charente-Maritime", 18: "Cher", 19: "Correze",
    201: "Corse-du-Sud", 202: "Haute-Corse",
    21: "Cote-d'Or", 22: "Cotes-d'Armor", 23: "Creuse", 24: "Dordogne",
    25: "Doubs", 26: "Drome", 27: "Eure", 28: "Eure-et-Loir",
    29: "Finistere", 30: "Gard", 31: "Haute-Garonne", 32: "Gers",
    33: "Gironde", 34: "Herault", 35: "Ille-et-Vilaine", 36: "Indre",
    37: "Indre-et-Loire", 38: "Isere", 39: "Jura", 40: "Landes",
    41: "Loir-et-Cher", 42: "Loire", 43: "Haute-Loire", 44: "Loire-Atlantique",
    45: "Loiret", 46: "Lot", 47: "Lot-et-Garonne", 48: "Lozere",
    49: "Maine-et-Loire", 50: "Manche", 51: "Marne", 52: "Haute-Marne",
    53: "Mayenne", 54: "Meurthe-et-Moselle", 55: "Meuse", 56: "Morbihan",
    57: "Moselle", 58: "Nievre", 59: "Nord", 60: "Oise",
    61: "Orne", 62: "Pas-de-Calais", 63: "Puy-de-Dome", 64: "Pyrenees-Atlantiques",
    65: "Hautes-Pyrenees", 66: "Pyrenees-Orientales", 67: "Bas-Rhin", 68: "Haut-Rhin",
    69: "Rhone", 70: "Haute-Saone", 71: "Saone-et-Loire", 72: "Sarthe",
    73: "Savoie", 74: "Haute-Savoie", 75: "Paris", 76: "Seine-Maritime",
    77: "Seine-et-Marne", 78: "Yvelines", 79: "Deux-Sevres", 80: "Somme",
    81: "Tarn", 82: "Tarn-et-Garonne", 83: "Var", 84: "Vaucluse",
    85: "Vendee", 86: "Vienne", 87: "Haute-Vienne", 88: "Vosges",
    89: "Yonne", 90: "Territoire de Belfort",
    91: "Essonne", 92: "Hauts-de-Seine", 93: "Seine-Saint-Denis",
    94: "Val-de-Marne", 95: "Val-d'Oise",
    # DOM
    971: "Guadeloupe", 972: "Martinique", 973: "Guyane",
    974: "La Reunion", 976: "Mayotte",
}


# Reverse lookup: department -> region (built once at import time)
_DEPT_TO_REGION: dict[int, str] = {
    dept: region
    for region, depts in REGIONS.items()
    for dept in depts
}


# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------

def department_to_region(department: int) -> str:
    """Return the region name for a given department number.

    Parameters
    ----------
    department:
        Department number (e.g. 29 for Finistere).

    Returns
    -------
    str
        Region name.

    Raises
    ------
    KeyError
        If the department number is not in the reference data.
    """
    try:
        return _DEPT_TO_REGION[department]
    except KeyError:
        raise KeyError(
            f"Unknown department number: {department}. "
            f"Valid numbers: {sorted(_DEPT_TO_REGION.keys())}"
        ) from None


def department_name(department: int) -> str:
    """Return the name of a French department given its number.

    Raises
    ------
    KeyError
        If the department number is not in the reference data.
    """
    try:
        return DEPARTMENTS[department]
    except KeyError:
        raise KeyError(f"Unknown department number: {department}") from None


def departments_in_region(region: str) -> list[int]:
    """Return the sorted list of department numbers in a region.

    Raises
    ------
    KeyError
        If the region name does not match any entry in :data:`REGIONS`.
    """
    try:
        return sorted(REGIONS[region])
    except KeyError:
        raise KeyError(
            f"Unknown region: {region!r}. "
            f"Valid regions: {sorted(REGIONS.keys())}"
        ) from None


def communes_in_department(department: int) -> list[dict]:
    """Return a list of communes in a department.

    .. note::

        This is a placeholder that returns an empty list.  A full
        implementation would call the INSEE COG (Code Officiel
        Geographique) API or read the local COG CSV file::

            https://www.insee.fr/fr/information/6800675

    Parameters
    ----------
    department:
        Department number.

    Returns
    -------
    list[dict]
        Each dict contains ``"insee_code"`` (int), ``"name"`` (str),
        and ``"postal_codes"`` (list[str]).  Currently returns ``[]``.
    """
    # Structure for future implementation:
    # url = f"https://geo.api.gouv.fr/departements/{dept_str}/communes"
    # resp = httpx.get(url, params={"fields": "nom,code,codesPostaux"})
    # return [{"insee_code": int(c["code"]), "name": c["nom"],
    #          "postal_codes": c["codesPostaux"]} for c in resp.json()]
    return []
