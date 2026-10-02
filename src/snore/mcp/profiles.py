"""Clinical profile presets for SNORE MCP.

Profiles shape the INSTRUCTIONS resource and suggested-priority hints only (G1).
No tool returns different *data* per profile — thresholds and severity ladders
live here in the instructions text, not in tool response logic.

Profiles available:
    neutral (default) — no clinical framing; reports all indices equally.
    uars — de-emphasizes AHI; leads with flow morphology and RERA/RDI.
    osa  — AHI-forward; emphasises obstructive event burden and compliance.
    csa  — leads with MV, periodic breathing, and central event characterization.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ClinicalProfile:
    name: str
    display_name: str
    priority_hint: str
    clinical_context: str


_PROFILES: dict[str, ClinicalProfile] = {
    "neutral": ClinicalProfile(
        name="neutral",
        display_name="Neutral",
        priority_hint="Report all indices (AHI, RDI, flow-limitation, leak, pressure, MV) equally.",
        clinical_context=(
            "No clinical framing is active. Interpret indices in the context of the "
            "dataset; do not apply population-level severity ladders without "
            "user-supplied thresholds."
        ),
    ),
    "uars": ClinicalProfile(
        name="uars",
        display_name="UARS (Upper Airway Resistance Syndrome)",
        priority_hint=(
            "De-emphasize AHI. Lead with flow morphology (flattening index, FL runs, "
            "RERA count/RDI) and inspiratory effort markers. Treat RDI > threshold as "
            "the primary burden index; treat AHI < 5 as consistent with UARS phenotype, "
            "not as 'normal'. Pressure tuning goal: eliminate flow-limited breaths while "
            "minimising leak. The RERA proxy and the RDI built on it are experimental "
            "SNORE heuristics, not device-scored events: present them as trends."
        ),
        clinical_context=(
            "UARS phenotype: RDI > 30, AHI < 5, inspiratory flow morphology is the "
            "primary signal. Flattening index and FL-run-ending-in-recovery-breath "
            "(RERA proxy) outrank AHI as tuning targets. "
            "Bilevel therapy (VAuto/ASV) context: IPAP drives upper-airway dilation; "
            "EPAP provides baseline support; PS = IPAP − EPAP. "
            "Thresholds used in this dataset are user-configured — do not apply "
            "generic AHI severity labels."
        ),
    ),
    "osa": ClinicalProfile(
        name="osa",
        display_name="OSA (Obstructive Sleep Apnea)",
        priority_hint=(
            "AHI-forward. Report OAI, CAI, HI, AHI as the primary burden. "
            "Compliance (≥4 h/night) is a key secondary metric. "
            "Pressure titration goal: suppress obstructive events and reduce AHI."
        ),
        clinical_context=(
            "OSA therapy context: primary goal is AHI suppression via adequate "
            "pressure. Compliance tracking matters for insurance and efficacy. "
            "Do not infer severity from AHI alone — report all components "
            "(OAI, CAI, HI) and let the user interpret."
        ),
    ),
    "csa": ClinicalProfile(
        name="csa",
        display_name="CSA / Periodic Breathing",
        priority_hint=(
            "Lead with MV, periodic-breathing percentage, and central event burden "
            "(CAI). Report MV rolling variance and respiratory rate stability as "
            "primary signals. Suppress back-up rate discussion unless the device "
            "reports it."
        ),
        clinical_context=(
            "CSA / complex sleep apnea context: central events and periodic breathing "
            "dominate. MV stability and respiratory rate regularity are the primary "
            "tuning signals. Flow morphology is secondary. "
            "Do not conflate CAI with OAI — report them separately."
        ),
    ),
}

VALID_PROFILES: frozenset[str] = frozenset(_PROFILES)


def get_profile(name: str) -> ClinicalProfile:
    """Return the named profile or raise ValueError for unknown names."""
    if name not in _PROFILES:
        raise ValueError(
            f"Unknown clinical profile {name!r}. "
            f"Valid profiles: {sorted(VALID_PROFILES)}"
        )
    return _PROFILES[name]


def list_profiles() -> list[ClinicalProfile]:
    """Return all profiles in a stable order."""
    return [_PROFILES[k] for k in ("neutral", "uars", "osa", "csa")]
