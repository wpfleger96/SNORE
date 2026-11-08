import type { AnalysisMode } from '@/api/analysis'

/** Display labels for the detection modes; key order is the canonical mode order. */
export const MODE_LABELS = {
    aasm: 'AASM',
    aasm_relaxed: 'AASM Relaxed',
    resmed: 'ResMed',
} satisfies Record<AnalysisMode, string>

export const ANALYSIS_MODES = Object.keys(MODE_LABELS) as AnalysisMode[]

/** The label for a mode key; keys the UI does not know show as-is. */
export function modeLabel(mode: string): string {
    return Object.hasOwn(MODE_LABELS, mode) ? MODE_LABELS[mode as AnalysisMode] : mode
}

/**
 * The primary mode a run uses: the explicit pick while it is selected, else
 * AASM when selected (the server's default), else the first selected mode in
 * canonical order. Undefined only when no mode is selected.
 */
export function effectivePrimaryMode(
    modes: readonly AnalysisMode[],
    pick: AnalysisMode | null,
): AnalysisMode | undefined {
    if (pick && modes.includes(pick)) return pick
    if (modes.includes('aasm')) return 'aasm'
    return ANALYSIS_MODES.find((m) => modes.includes(m))
}
