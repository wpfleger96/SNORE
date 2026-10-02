/** Shared date/time, quantity, and classification formatting helpers. */

import { ahiTier } from '@/utils/ahiScale'

export function ahiClass(ahi: number | null | undefined): string {
    const tier = ahiTier(ahi)
    return tier ? `ahi-${tier}` : ''
}

/** e.g. parseLocalDate("2026-05-28") — treats date-only strings as local midnight to avoid UTC shift. */
export function parseLocalDate(iso: string): Date {
    const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso)
    if (m) return new Date(Number(m[1]), Number(m[2]) - 1, Number(m[3]))
    return new Date(iso)
}

/** e.g. "Jan 5, 03:12 AM" — short date with time, no year. */
export function formatDateShort(iso: string): string {
    return parseLocalDate(iso).toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    })
}

/** e.g. "Jan 5, 2024" — full date, no time. */
export function formatDateFull(iso: string): string {
    return parseLocalDate(iso).toLocaleDateString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
    })
}

/** e.g. "Fri, Jan 5, 2024" — full date with weekday. */
export function formatDateWithWeekday(iso: string): string {
    return parseLocalDate(iso).toLocaleDateString(undefined, {
        weekday: 'short',
        year: 'numeric',
        month: 'short',
        day: 'numeric',
    })
}

/** e.g. "Jan 5, 2024, 03:12 AM" — full date with time. */
export function formatDateTime(iso: string): string {
    return parseLocalDate(iso).toLocaleString(undefined, {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    })
}

/** e.g. "Jan 5" — month and day only. */
export function formatDateMonthDay(iso: string): string {
    return parseLocalDate(iso).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
}

/** e.g. "just now", "5m ago", "3h ago", or "Jan 5, 03:12 AM" for older times.
 *  Accepts an ISO 8601 datetime string; returns "—" for invalid input. */
export function formatRelativeTime(value: string): string {
    const d = new Date(value)
    if (isNaN(d.getTime())) return '—'
    const diffMin = Math.floor((Date.now() - d.getTime()) / 60_000)
    if (diffMin < 1) return 'just now'
    if (diffMin < 60) return `${diffMin}m ago`
    const diffH = Math.floor(diffMin / 60)
    if (diffH < 24) return `${diffH}h ago`
    return d.toLocaleDateString(undefined, {
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
    })
}

/** e.g. "2024-01-05" — ISO calendar date in local time. */
export function formatIso(d: Date): string {
    const y = d.getFullYear()
    const m = String(d.getMonth() + 1).padStart(2, '0')
    const day = String(d.getDate()).padStart(2, '0')
    return `${y}-${m}-${day}`
}

/** e.g. "1:02:03" — clock-style h:mm:ss offset from a duration in seconds. */
export function formatTimeOffset(secs: number): string {
    const h = Math.floor(secs / 3600)
    const m = Math.floor((secs % 3600) / 60)
    const s = Math.floor(secs % 60)
    return `${h}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

/** e.g. "325.5 KB", "1.2 GB" — human-readable byte count. */
export function formatBytes(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(1)} GB`
}

// Adaptive percent formatting, mirroring the backend's `fmt_sig` philosophy: fixed
// decimals at ordinary magnitudes, but significant figures / scientific notation for
// tiny values so a chance-precision floor (~4e-5 → 0.004%) stays legible beside a
// proxy precision (~1e-3 → 0.1%) instead of both flooring to 0.00%. Comparing those
// near-zero magnitudes is the whole point of the RERA framing.
function adaptivePercentText(pct: number, decimals: number): string {
    const a = Math.abs(pct)
    if (a === 0 || a >= 0.1) return pct.toFixed(decimals)
    if (a >= 1e-4) return pct.toPrecision(2)
    return pct.toExponential(1)
}

/** Format a 0–1 ratio as an adaptive-precision percent string (e.g. "25.0%",
 *  "0.0040%"); null when the input is null/undefined/non-finite. */
export function formatPercent(ratio: number | null | undefined, decimals = 1): string | null {
    if (ratio == null || !Number.isFinite(ratio)) return null
    return `${adaptivePercentText(ratio * 100, decimals)}%`
}

/** Format a signed ratio delta as adaptive-precision percentage points (e.g.
 *  "+5.0 pp", "+0.0040 pp"); null when the input is null/undefined/non-finite. */
export function formatPercentPointsDelta(
    delta: number | null | undefined,
    decimals = 1,
): string | null {
    if (delta == null || !Number.isFinite(delta)) return null
    const pp = delta * 100
    const sign = pp > 0 ? '+' : ''
    return `${sign}${adaptivePercentText(pp, decimals)} pp`
}

// Lazily-created, module-cached formatters for uPlot axis tick labels (allocation-free on hot path).
let _fmtWithSecs: Intl.DateTimeFormat | undefined
let _fmtNoSecs: Intl.DateTimeFormat | undefined

/** e.g. "10:30 PM" or "10:30:15 PM" — wall-clock time for uPlot axis ticks.
 *  Shows seconds when the tick increment is finer than 1 minute (`foundIncr < 60`). */
export function formatWallClockTime(epochSecs: number, foundIncr: number): string {
    const d = new Date(epochSecs * 1000)
    if (foundIncr < 60) {
        _fmtWithSecs ??= new Intl.DateTimeFormat(undefined, {
            hour: 'numeric',
            minute: '2-digit',
            second: '2-digit',
        })
        return _fmtWithSecs.format(d)
    }
    _fmtNoSecs ??= new Intl.DateTimeFormat(undefined, {
        hour: 'numeric',
        minute: '2-digit',
    })
    return _fmtNoSecs.format(d)
}

/** Convert seconds to fractional hours; returns null when the input is null or undefined. */
export function secToHours(sec: number | null | undefined): number | null {
    return sec != null ? sec / 3600 : null
}

/** Average of a nullable array; returns null when there are no non-null values. */
export function avg(vals: (number | null | undefined)[]): number | null {
    const nonNull = vals.filter((v): v is number => v != null)
    return nonNull.length ? nonNull.reduce((a, b) => a + b, 0) / nonNull.length : null
}

// Friendly text for the backend's null-with-reason codes (NullReason enum), used
// to explain why an experimental metric is absent for a given night. Unmapped
// codes fall back to the prettified code so new reasons stay readable.
const NULL_REASON_LABELS: Record<string, string> = {
    analysis_not_run: 'Breath analysis has not been run for this night.',
    analysis_stale: 'Breath analysis is out of date for this night.',
    device_ambiguous:
        'The night has sessions from more than one device and no device was pinned to disambiguate.',
    algo_version_mismatch: 'Produced by a different analysis algorithm version.',
    channel_absent: 'A required signal channel is not available.',
    channel_unaligned: 'Signal channels could not be aligned for this night.',
    not_available: 'Not available for this night.',
    no_data_in_range: 'No breath data was recorded for this night.',
    table_missing: 'Breath-level data has not been stored for this night.',
    duration_zero: 'Therapy duration was zero, so a per-hour rate is undefined.',
    no_sessions: 'No therapy sessions were recorded for this night.',
    primary_mode_mismatch:
        "This night's therapy mode did not match the primary mode for this period.",
    smart_ramp_indeterminate:
        'The smart-ramp period could not be determined, so sleep onset could not be excluded.',
    segments_unknown: 'The breath segments for this night could not be determined.',
    multi_session_ambiguity:
        'Multiple overlapping sessions made this metric ambiguous for the night.',
    unvalidated_device: 'This device model has not been validated for this metric.',
    rx_changed_within_epoch: 'The prescription changed partway through this period.',
}

/** Humanize a NullReason code into a sentence for tooltips; null when absent. */
export function nullReasonLabel(reason: string | null | undefined): string | null {
    if (!reason) return null
    return NULL_REASON_LABELS[reason] ?? reason.replace(/_/g, ' ')
}
