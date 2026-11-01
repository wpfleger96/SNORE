"""Clinical profile presets for SNORE MCP.

Profiles shape the INSTRUCTIONS resource and suggested-priority hints only (G1).
No tool returns different *data* per profile — thresholds and severity ladders
live here in the instructions text, not in tool response logic.

Profiles available:
    neutral (default) — no clinical framing; reports device indices equally.
    uars — de-emphasizes AHI; leads with device RE counts, the device
        flow-limitation channel, and flow-waveform inspection.
    osa  — AHI-forward; emphasises obstructive event burden and compliance.
    csa  — leads with device central event counts, MV, and respiratory rate.

Every profile leads with fields on the default tools.  ``experimental_*`` tools
appear only as optional trend evidence.
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
        priority_hint="Report all indices (AHI, device flow-limitation channel, leak, pressure, MV) equally.",
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
            "De-emphasize AHI. Lead with the device RE count (reras), the device "
            "flow-limitation channel (device_flg_*), and inspection of the flow "
            "waveform (render_window, get_waveform) for inspiratory flattening. Treat "
            "AHI < 5 as consistent with UARS phenotype, not as 'normal'. Pressure "
            "tuning goal: reduce device flow limitation and RE events while minimising "
            "leak. experimental_get_nightly_metrics is optional trend evidence: its "
            "RERA proxy, the RDI built on it, and its flattening metrics are SNORE's "
            "own heuristics (experimental trend instruments, not device-scored or "
            "clinically validated), so present them only as trends."
        ),
        clinical_context=(
            "UARS phenotype: AHI < 5 with inspiratory flow morphology as the primary "
            "signal. Device RE events, the device flow-limitation channel, and "
            "flattened inspiration on the flow waveform outrank AHI as tuning targets. "
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
            "Lead with device central event burden (central_apneas, CAI), then device "
            "MV (mv_mean_lpm, mv waveform) and respiratory rate (rr_mean_bpm, rr "
            "waveform). Suppress back-up rate discussion unless the device reports it. "
            "experimental_get_nightly_metrics (periodic-breathing percentage) and "
            "experimental_get_event_context (MV slope and stability before events) "
            "are optional trend evidence, not device-scored or clinically validated."
        ),
        clinical_context=(
            "CSA / complex sleep apnea context: central events and periodic breathing "
            "dominate. Device MV and respiratory rate traces around central events are "
            "the primary tuning signals. Flow morphology is secondary. "
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
