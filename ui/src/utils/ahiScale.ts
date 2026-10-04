export type AhiTier = 'good' | 'mild' | 'moderate' | 'severe'

export interface AhiScaleEntry {
    tier: AhiTier
    label: string
    color: string // cell/legend fill; must match CalendarHeatmap's `.cell--*` backgrounds
    maxAhi: number | null // null = no upper bound (catch-all)
}

// Shared AHI colour scale: the single source of severity cut points for the
// CalendarHeatmap cells, the DashboardView legend, and the `ahi-*` text classes.
// Thresholds: <5 good, 5–9 mild, 10–14 moderate, ≥15 severe.
// Note: this display scale is stricter than the common clinical convention
// (<5 normal, 5–15 mild, 15–30 moderate, >30 severe).
export const AHI_COLOR_SCALE: AhiScaleEntry[] = [
    { tier: 'good', label: 'AHI < 5 — Good', color: '#22c55e', maxAhi: 5 },
    { tier: 'mild', label: 'AHI 5–9 — Mild', color: '#eab308', maxAhi: 10 },
    { tier: 'moderate', label: 'AHI 10–14 — Moderate', color: '#f97316', maxAhi: 15 },
    { tier: 'severe', label: 'AHI ≥ 15 — Severe', color: '#ef4444', maxAhi: null },
]

export function ahiTier(ahi: number | null | undefined): AhiTier | null {
    if (ahi == null || Number.isNaN(ahi)) return null
    return AHI_COLOR_SCALE.find((e) => e.maxAhi == null || ahi < e.maxAhi)!.tier
}

export function ahiColorClass(ahi: number | null): string {
    const tier = ahiTier(ahi)
    return tier ? `cell--${tier}` : 'cell--empty'
}
