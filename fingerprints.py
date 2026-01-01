"""
fingerprints.py
Module for molecular feature extraction from SMILES strings
For use in Streamlit web server

Workflow:
    - User inputs raw SMILES
    - Convert to canonical SMILES (in app.py via canonicalize_smiles)
    - Compute fingerprints from canonical SMILES

Included features:
    - ECFP (Morgan fingerprint)
    - RDKit path-based fingerprint
    - MACCS keys
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys


# -------------------------------------------------------------------
# 0. Canonicalization
# -------------------------------------------------------------------
def canonicalize_smiles(smiles: str):
    """
    Convert a user-input SMILES string to RDKit canonical SMILES.

    Parameters
    ----------
    smiles : str
        Raw SMILES from user.

    Returns
    -------
    str or None
        Canonical SMILES string if valid, otherwise None.
    """
    if smiles is None:
        return None

    s = str(smiles).strip()
    if not s:
        return None

    try:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


# -------------------------------------------------------------------
# 1. ECFP Fingerprint (Morgan) – using training config
# -------------------------------------------------------------------
def get_ecfp(canonical_smiles: str, radius: int = 10, n_bits: int = 4096):
    """
    Calculate ECFP (Morgan) fingerprint from a *canonical* SMILES string.

    Parameters
    ----------
    canonical_smiles : str
        Canonical SMILES (already processed by canonicalize_smiles).
    radius : int, optional
        Morgan radius, by default 10 (match training).
    n_bits : int, optional
        Length of fingerprint, by default 4096 (match training).

    Returns
    -------
    np.ndarray or None
        1D numpy array of bits (0/1) or None if SMILES invalid.
    """
    if canonical_smiles is None:
        return None

    try:
        mol = Chem.MolFromSmiles(canonical_smiles)
        if mol is None:
            return None
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        return np.array(fp, dtype=np.int8)
    except Exception:
        return None


# -------------------------------------------------------------------
# 2. RDKit path-based Fingerprint (2048 bits)
# -------------------------------------------------------------------
def get_rdkit(canonical_smiles: str, n_bits: int = 2048):
    """
    Calculate RDKit path-based fingerprint from a *canonical* SMILES string.

    Parameters
    ----------
    canonical_smiles : str
        Canonical SMILES (already processed by canonicalize_smiles).
    n_bits : int, optional
        Length of fingerprint (fpSize), by default 2048 (match training).

    Returns
    -------
    np.ndarray or None
        1D numpy array of bits (0/1) or None if SMILES invalid.
    """
    if canonical_smiles is None:
        return None

    try:
        mol = Chem.MolFromSmiles(canonical_smiles)
        if mol is None:
            return None
        fp = Chem.RDKFingerprint(mol, fpSize=n_bits)
        return np.array(fp, dtype=np.int8)
    except Exception:
        return None


# -------------------------------------------------------------------
# 3. MACCS Keys (167 bits)
# -------------------------------------------------------------------
def get_maccs(canonical_smiles: str):
    """
    Calculate MACCS keys fingerprint (167 bits) from a *canonical* SMILES string.

    Parameters
    ----------
    canonical_smiles : str
        Canonical SMILES (already processed by canonicalize_smiles).

    Returns
    -------
    np.ndarray or None
        1D numpy array of bits (0/1) of length 167,
        or None if SMILES invalid.
    """
    if canonical_smiles is None:
        return None

    try:
        mol = Chem.MolFromSmiles(canonical_smiles)
        if mol is None:
            return None
        fp = MACCSkeys.GenMACCSKeys(mol)
        return np.array(fp, dtype=np.int8)
    except Exception:
        return None


# -------------------------------------------------------------------
# Helper: convert fingerprint to float32 (for model input)
# -------------------------------------------------------------------
def safe_fp(fp):
    """
    Convert fingerprint to float32 numpy array.
    Return None if fp is None.
    """
    return np.array(fp, dtype=np.float32) if fp is not None else None
