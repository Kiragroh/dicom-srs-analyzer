"""
SRS Analysis Pipeline - Anonymization Utilities
================================================
Provides functions to anonymize patient IDs and structure names for public sharing.
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

# Global mapping caches (persistent across calls within a single run)
_patient_id_map: Dict[str, str] = {}
_structure_name_map: Dict[str, str] = {}
_patient_counter = 0
_structure_counter = 0


def reset_anonymization():
    """Reset all anonymization mappings (useful for testing)."""
    global _patient_id_map, _structure_name_map, _patient_counter, _structure_counter
    _patient_id_map.clear()
    _structure_name_map.clear()
    _patient_counter = 0
    _structure_counter = 0


def anonymize_patient_id(patient_id: str) -> str:
    """
    Anonymize a patient ID to Patient_001, Patient_002, etc.
    
    Maintains consistent mapping within a single pipeline run.
    """
    global _patient_counter
    
    if patient_id not in _patient_id_map:
        _patient_counter += 1
        _patient_id_map[patient_id] = f"Patient_{_patient_counter:03d}"
    
    return _patient_id_map[patient_id]


def anonymize_structure_name(structure_name: str) -> str:
    """
    Anonymize a structure name to PTV_A, PTV_B, etc.
    
    Maintains consistent mapping within a single pipeline run.
    Non-PTV structures are passed through unchanged.
    """
    global _structure_counter
    
    # Only anonymize PTV structures
    if not structure_name.upper().startswith("PTV"):
        return structure_name
    
    if structure_name not in _structure_name_map:
        _structure_counter += 1
        # Use letters A-Z, then AA, AB, etc.
        if _structure_counter <= 26:
            suffix = chr(64 + _structure_counter)  # A=65, B=66, ...
        else:
            first = chr(64 + ((_structure_counter - 1) // 26))
            second = chr(65 + ((_structure_counter - 1) % 26))
            suffix = f"{first}{second}"
        
        _structure_name_map[structure_name] = f"PTV_{suffix}"
    
    return _structure_name_map[structure_name]


def get_anonymization_mapping() -> Dict[str, Dict[str, str]]:
    """
    Return the current anonymization mappings for logging/debugging.
    
    Returns:
        dict with keys 'patients' and 'structures', each mapping original -> anonymized
    """
    return {
        "patients": _patient_id_map.copy(),
        "structures": _structure_name_map.copy(),
    }
