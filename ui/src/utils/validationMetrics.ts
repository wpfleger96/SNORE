/** Aggregate-metric descriptors and run-comparison helpers.
 *
 * The comparison view is validator-agnostic: each validator type declares which
 * aggregate fields matter, how to read them (dotted path into the report's
 * `aggregate`), how to format them, and whether a higher value is "better". The
 * delta and identity-diff functions below are pure so they can be unit-tested. */
import type { ValidatorType } from '@/types'
import { formatPercent, formatPercentPointsDelta } from '@/utils/formatting'
import { provenanceFor, type Provenance } from '@/utils/provenance'

export const VALIDATOR_LABELS: Record<ValidatorType, string> = {
    events: 'Events',
    fl: 'FL vs FLG',
    breaths: 'Breath trends',
    rera: 'RERA',
    apple: 'Apple cross',
}

// percent: 0–1 ratio rendered as %. decimal: signed correlation/AUC (-1..1).
// rate: events per hour. count: integer tallies.
export type MetricKind = 'percent' | 'decimal' | 'rate' | 'count'

export interface MetricDescriptor {
    // Dotted path into `report.aggregate` (e.g. 'rr.mean_spearman_r').
    path: string
    label: string
    kind: MetricKind
    glossaryKey?: string
    // Direction that counts as an improvement; omitted when neutral (counts).
    higherIsBetter?: boolean
    // Tier of the value, resolved from the backend's tag for the path's leaf field.
    provenance: Provenance
}

type MetricSpec = Omit<MetricDescriptor, 'provenance'>

function withProvenance(
    specs: Record<ValidatorType, MetricSpec[]>,
): Record<ValidatorType, MetricDescriptor[]> {
    const out = {} as Record<ValidatorType, MetricDescriptor[]>
    for (const [type, metrics] of Object.entries(specs) as [ValidatorType, MetricSpec[]][]) {
        out[type] = metrics.map((m) => ({
            ...m,
            provenance: provenanceFor(m.path.slice(m.path.lastIndexOf('.') + 1)),
        }))
    }
    return out
}

export const AGGREGATE_METRICS = withProvenance({
    events: [
        {
            path: 'avg_apnea_sensitivity',
            label: 'Apnea Sensitivity',
            kind: 'percent',
            glossaryKey: 'sensitivity',
            higherIsBetter: true,
        },
        {
            path: 'avg_apnea_precision',
            label: 'Apnea Precision',
            kind: 'percent',
            glossaryKey: 'precision',
            higherIsBetter: true,
        },
        {
            path: 'avg_apnea_f1',
            label: 'Apnea F1',
            kind: 'percent',
            glossaryKey: 'f1',
            higherIsBetter: true,
        },
        {
            path: 'avg_hypopnea_sensitivity',
            label: 'Hypopnea Sensitivity',
            kind: 'percent',
            glossaryKey: 'sensitivity',
            higherIsBetter: true,
        },
        {
            path: 'avg_hypopnea_precision',
            label: 'Hypopnea Precision',
            kind: 'percent',
            glossaryKey: 'precision',
            higherIsBetter: true,
        },
        {
            path: 'avg_hypopnea_f1',
            label: 'Hypopnea F1',
            kind: 'percent',
            glossaryKey: 'f1',
            higherIsBetter: true,
        },
        { path: 'total_sessions', label: 'Sessions', kind: 'count' },
    ],
    fl: [
        {
            path: 'mean_spearman_flattening_r',
            label: 'Spearman (Flattening)',
            kind: 'decimal',
            glossaryKey: 'spearman_r',
            higherIsBetter: true,
        },
        {
            path: 'mean_spearman_class_weight_r',
            label: 'Spearman (Class Weight)',
            kind: 'decimal',
            glossaryKey: 'spearman_r',
            higherIsBetter: true,
        },
        {
            path: 'mean_auc_t25',
            label: 'AUC25',
            kind: 'decimal',
            glossaryKey: 'auc',
            higherIsBetter: true,
        },
        {
            path: 'mean_auc_t50',
            label: 'AUC50',
            kind: 'decimal',
            glossaryKey: 'auc',
            higherIsBetter: true,
        },
        {
            path: 'mean_auc_class_t25',
            label: 'Class AUC25',
            kind: 'decimal',
            glossaryKey: 'auc',
            higherIsBetter: true,
        },
        {
            path: 'mean_auc_class_t50',
            label: 'Class AUC50',
            kind: 'decimal',
            glossaryKey: 'auc',
            higherIsBetter: true,
        },
        {
            path: 'cross_night_spearman_r',
            label: 'Cross-night Spearman',
            kind: 'decimal',
            glossaryKey: 'cross_night_spearman',
            higherIsBetter: true,
        },
        { path: 'sessions_compared', label: 'Sessions Compared', kind: 'count' },
    ],
    breaths: [
        {
            path: 'rr.mean_spearman_r',
            label: 'RR Spearman',
            kind: 'decimal',
            glossaryKey: 'spearman_r',
            higherIsBetter: true,
        },
        {
            path: 'tv.mean_spearman_r',
            label: 'TV Spearman',
            kind: 'decimal',
            glossaryKey: 'spearman_r',
            higherIsBetter: true,
        },
        {
            path: 'ti.mean_spearman_r',
            label: 'Ti Spearman',
            kind: 'decimal',
            glossaryKey: 'spearman_r',
            higherIsBetter: true,
        },
        {
            path: 'ie_ratio.mean_spearman_r',
            label: 'I:E Spearman',
            kind: 'decimal',
            glossaryKey: 'spearman_r',
            higherIsBetter: true,
        },
        { path: 'sessions_compared', label: 'Sessions Compared', kind: 'count' },
    ],
    rera: [
        {
            path: 'mean_amplitude_sensitivity',
            label: 'Amplitude Sensitivity',
            kind: 'percent',
            glossaryKey: 'sensitivity',
            higherIsBetter: true,
        },
        {
            path: 'mean_amplitude_precision',
            label: 'Amplitude Precision',
            kind: 'percent',
            glossaryKey: 'precision',
            higherIsBetter: true,
        },
        {
            path: 'mean_proxy_sensitivity',
            label: 'Proxy Sensitivity',
            kind: 'percent',
            glossaryKey: 'sensitivity',
            higherIsBetter: true,
        },
        {
            path: 'mean_proxy_precision',
            label: 'Proxy Precision',
            kind: 'percent',
            glossaryKey: 'precision',
            higherIsBetter: true,
        },
        {
            path: 'chance_precision_floor',
            label: 'Chance Precision Floor',
            kind: 'percent',
            glossaryKey: 'chance_floor',
        },
        { path: 'proxy_density', label: 'Proxy Density (/h)', kind: 'rate' },
        { path: 'machine_re_density', label: 'Device RE Density (/h)', kind: 'rate' },
        { path: 'total_proxy_reras', label: 'Total Proxy RERAs', kind: 'count' },
    ],
    apple: [
        {
            path: 'rera_vs_apple_bd.rho',
            label: 'RERA vs Apple BD',
            kind: 'decimal',
            glossaryKey: 'apple_breathing_disturbances',
            higherIsBetter: true,
        },
        {
            path: 'fl_vs_apple_bd.rho',
            label: 'FL vs Apple BD',
            kind: 'decimal',
            glossaryKey: 'apple_breathing_disturbances',
            higherIsBetter: true,
        },
        {
            path: 'rera_vs_awake_seconds.rho',
            label: 'RERA vs Awake Time',
            kind: 'decimal',
            glossaryKey: 'spearman_r',
            higherIsBetter: true,
        },
        {
            path: 'fl_vs_sleep_efficiency.rho',
            label: 'FL vs Sleep Efficiency',
            kind: 'decimal',
            glossaryKey: 'spearman_r',
        },
        { path: 'n_with_apple_bd', label: 'Nights with Apple BD', kind: 'count' },
        { path: 'total_nights', label: 'Total Nights', kind: 'count' },
    ],
})

/** Read a possibly-nested numeric field by dotted path; null when absent/non-numeric. */
export function getByPath(obj: unknown, path: string): number | null {
    let cur: unknown = obj
    for (const key of path.split('.')) {
        if (cur == null || typeof cur !== 'object') return null
        cur = (cur as Record<string, unknown>)[key]
    }
    return typeof cur === 'number' && Number.isFinite(cur) ? cur : null
}

export function formatMetric(value: number | null, kind: MetricKind): string {
    if (value == null) return '—'
    switch (kind) {
        case 'percent':
            return formatPercent(value) ?? '—'
        case 'decimal':
            return value.toFixed(3)
        case 'rate':
            return value.toFixed(2)
        case 'count':
            return String(value)
    }
}

export interface MetricDelta {
    a: number | null
    b: number | null
    delta: number | null // b - a, null when either side is null
    // 'better' | 'worse' | 'neutral' — relative to the metric's higherIsBetter,
    // 'neutral' when direction is undefined or the delta is zero/unknown.
    direction: 'better' | 'worse' | 'neutral'
}

/** Compare two aggregates for one metric: raw values, signed delta (b − a), and
 *  whether b improved on a given the metric's preferred direction. */
export function computeDelta(
    aggregateA: unknown,
    aggregateB: unknown,
    metric: MetricDescriptor,
): MetricDelta {
    const a = getByPath(aggregateA, metric.path)
    const b = getByPath(aggregateB, metric.path)
    if (a == null || b == null) {
        return { a, b, delta: null, direction: 'neutral' }
    }
    const delta = b - a
    let direction: MetricDelta['direction'] = 'neutral'
    if (metric.higherIsBetter !== undefined && delta !== 0) {
        const improved = metric.higherIsBetter ? delta > 0 : delta < 0
        direction = improved ? 'better' : 'worse'
    }
    return { a, b, delta, direction }
}

export function formatDelta(delta: number | null, kind: MetricKind): string {
    if (delta == null) return '—'
    if (kind === 'percent') return formatPercentPointsDelta(delta) ?? '—'
    const sign = delta > 0 ? '+' : ''
    if (kind === 'count') return `${sign}${delta}`
    return `${sign}${delta.toFixed(kind === 'rate' ? 2 : 3)}`
}

/** Flatten an engine-identity or params object into stable `key value` chips. */
export function identityChips(identity: Record<string, unknown> | null | undefined): string[] {
    if (!identity) return []
    return Object.keys(identity)
        .sort()
        .map((k) => `${k} ${String(identity[k])}`)
}

/** Keys whose values differ between two identity/params objects (union of keys). */
export function diffKeys(
    a: Record<string, unknown> | null | undefined,
    b: Record<string, unknown> | null | undefined,
): Set<string> {
    const out = new Set<string>()
    const keys = new Set([...Object.keys(a ?? {}), ...Object.keys(b ?? {})])
    for (const k of keys) {
        if (JSON.stringify(a?.[k]) !== JSON.stringify(b?.[k])) out.add(k)
    }
    return out
}
