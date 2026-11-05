"""
Therapy-setting value labels shared across parsers.

Codes are on the ResMed S9/AirSense 10 basis, which OSCAR also stores after
normalizing AirSense 11 values, so the ResMed EDF and OSCAR-binary parsers
label the same setting identically.
"""

__all__ = ["CYCLE_MAP", "EPR_TYPE_MAP", "TRIGGER_MAP"]

# EPR type (OSCAR resmed_loader.cpp RMS9_EPR options).
EPR_TYPE_MAP: dict[int, str] = {0: "Off", 1: "Ramp Only", 2: "Full Time"}
# Trigger/cycle sensitivity (OSCAR resmed_loader.cpp :288-294 Trigger,
# :280-286 Cycle). "Med" spelling matches OSCAR.
TRIGGER_MAP: dict[int, str] = {
    0: "Very Low",
    1: "Low",
    2: "Med",
    3: "High",
    4: "Very High",
}
CYCLE_MAP: dict[int, str] = TRIGGER_MAP  # Same five-step sensitivity scale
