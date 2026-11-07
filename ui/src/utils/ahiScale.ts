export type AhiTier = 'good' | 'mild' | 'moderate' | 'severe'

export interface AhiScaleEntry {
    tier: AhiTier
    label: string
    color: string // legend swatch fill; the same `--ahi-*-fill` token as CalendarHeatmap's `.cell--*`
    maxAhi: number | null // null = no upper bound (catch-all)
}

// Shared AHI colour scale: the single source of severity cut points for the
// CalendarHeatmap cells, the DashboardView legend, and the `ahi-*` text classes.
// Colours live in tailwind.css as `--ahi-<tier>` (text) and `--ahi-<tier>-fill`.
// Thresholds: <5 good, 5–9 mild, 10–14 moderate, ≥15 severe.
// Note: this display scale is stricter than the common clinical convention
// (<5 normal, 5–15 mild, 15–30 moderate, >30 severe).
export const AHI_COLOR_SCALE: AhiScaleEntry[] = [
    { tier: 'good', label: 'AHI < 5 — Good', color: 'var(--ahi-good-fill)', maxAhi: 5 },
    { tier: 'mild', label: 'AHI 5–9 — Mild', color: 'var(--ahi-mild-fill)', maxAhi: 10 },
    {
        tier: 'moderate',
        label: 'AHI 10–14 — Moderate',
        color: 'var(--ahi-moderate-fill)',
        maxAhi: 15,
    },
    { tier: 'severe', label: 'AHI ≥ 15 — Severe', color: 'var(--ahi-severe-fill)', maxAhi: null },
]

export function ahiTier(ahi: number | null | undefined): AhiTier | null {
    if (ahi == null || Number.isNaN(ahi)) return null
    return AHI_COLOR_SCALE.find((e) => e.maxAhi == null || ahi < e.maxAhi)!.tier
}

export function ahiColorClass(ahi: number | null): string {
    const tier = ahiTier(ahi)
    return tier ? `cell--${tier}` : 'cell--empty'
}
