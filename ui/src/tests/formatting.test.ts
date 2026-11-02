// @vitest-environment node
import { describe, expect, it, vi, afterEach } from 'vitest'
import {
    formatDuration,
    formatPercent,
    formatPercentPointsDelta,
    formatRelativeTime,
    formatWallClockTime,
    nullReasonLabel,
} from '@/utils/formatting'

describe('formatDuration', () => {
    it('test_real_duration_shows_one_decimal_with_seconds_suffix', () => {
        expect(formatDuration(12.34)).toBe('12.3s')
    })

    it.each([null, undefined, 0])('test_absent_duration_%s_shows_em_dash', (value) => {
        expect(formatDuration(value)).toBe('—')
    })
})

describe('formatPercent', () => {
    it('test_ordinary_ratio_uses_fixed_decimals', () => {
        expect(formatPercent(0.25)).toBe('25.0%')
        expect(formatPercent(0.25, 2)).toBe('25.00%')
        expect(formatPercent(0.001, 2)).toBe('0.10%')
    })

    it('test_chance_floor_scale_stays_visible_not_zero', () => {
        // ~4e-5 ratio is the chance precision floor: it must NOT floor to 0.00%.
        expect(formatPercent(4e-5, 2)).toBe('0.0040%')
        // ~3.6e-4 scored floor.
        expect(formatPercent(3.6e-4, 2)).toBe('0.036%')
    })

    it('test_vanishing_magnitude_switches_to_scientific', () => {
        expect(formatPercent(5e-7)).toBe('5.0e-5%')
    })

    it('test_zero_and_nullish_and_non_finite', () => {
        expect(formatPercent(0)).toBe('0.0%')
        expect(formatPercent(null)).toBeNull()
        expect(formatPercent(undefined)).toBeNull()
        expect(formatPercent(Number.NaN)).toBeNull()
    })
})

describe('formatPercentPointsDelta', () => {
    it('test_ordinary_delta_signs_percentage_points', () => {
        expect(formatPercentPointsDelta(0.05)).toBe('+5.0 pp')
        expect(formatPercentPointsDelta(-0.05)).toBe('-5.0 pp')
        expect(formatPercentPointsDelta(0)).toBe('0.0 pp')
    })

    it('test_chance_floor_scale_delta_stays_visible', () => {
        // A run-vs-run move at the chance-floor scale must be representable, not +0.0 pp.
        expect(formatPercentPointsDelta(4e-5)).toBe('+0.0040 pp')
        expect(formatPercentPointsDelta(-4e-5)).toBe('-0.0040 pp')
    })

    it('test_nullish_returns_null', () => {
        expect(formatPercentPointsDelta(null)).toBeNull()
        expect(formatPercentPointsDelta(undefined)).toBeNull()
    })
})

describe('nullReasonLabel', () => {
    it('test_known_code_returns_sentence', () => {
        expect(nullReasonLabel('analysis_not_run')).toBe(
            'Breath analysis has not been run for this night.',
        )
    })

    it('test_unknown_code_prettifies_underscores', () => {
        expect(nullReasonLabel('some_future_reason')).toBe('some future reason')
    })

    // The six codes below were added alongside the FL/RERA day-view metrics; a
    // mapped code must resolve to its full sentence, never the prettify fallback.
    it.each([
        'primary_mode_mismatch',
        'smart_ramp_indeterminate',
        'segments_unknown',
        'multi_session_ambiguity',
        'unvalidated_device',
        'rx_changed_within_epoch',
    ])('test_mapped_code_%s_resolves_to_full_sentence', (code) => {
        const label = nullReasonLabel(code)
        expect(label).not.toBe(code.replace(/_/g, ' '))
        expect(label?.endsWith('.')).toBe(true)
    })

    it('test_nullish_reason_returns_null', () => {
        expect(nullReasonLabel(null)).toBeNull()
        expect(nullReasonLabel(undefined)).toBeNull()
        expect(nullReasonLabel('')).toBeNull()
    })
})

describe('formatRelativeTime', () => {
    const NOW = new Date(2026, 3, 6, 12, 0, 0)

    afterEach(() => {
        vi.useRealTimers()
    })

    function at(msBefore: number): string {
        return new Date(NOW.getTime() - msBefore).toISOString()
    }

    it('test_recent_string_renders_just_now', () => {
        vi.useFakeTimers()
        vi.setSystemTime(NOW)
        expect(formatRelativeTime(at(0))).toBe('just now')
    })

    it('test_five_minutes_ago_renders_minutes', () => {
        vi.useFakeTimers()
        vi.setSystemTime(NOW)
        expect(formatRelativeTime(at(5 * 60_000))).toBe('5m ago')
    })

    it('test_three_hours_ago_renders_hours', () => {
        vi.useFakeTimers()
        vi.setSystemTime(NOW)
        expect(formatRelativeTime(at(3 * 3_600_000))).toBe('3h ago')
    })

    it('test_older_than_24h_renders_localized_date', () => {
        vi.useFakeTimers()
        vi.setSystemTime(NOW)
        const result = formatRelativeTime(at(48 * 3_600_000))
        expect(result).not.toBe('just now')
        expect(result).not.toMatch(/\bago$/)
    })

    it('test_invalid_string_returns_em_dash', () => {
        expect(formatRelativeTime('not-a-date')).toBe('—')
    })
})

describe('formatWallClockTime', () => {
    // Build epoch via local-time constructor to avoid UTC/timezone dependency across machines.
    // The date component is irrelevant — only the time fields (h/m/s) are asserted.
    const epoch = new Date(2026, 3, 6, 22, 30, 15).getTime() / 1000 // 22:30:15 local

    it('test_below_60_incr_includes_seconds_in_output', () => {
        const result = formatWallClockTime(epoch, 30)
        expect(result).toContain('15')
    })

    it('test_below_60_incr_has_two_colon_separators', () => {
        const result = formatWallClockTime(epoch, 30)
        // h:mm:ss produces two colons regardless of locale 12/24h choice
        const colonCount = (result.match(/:/g) ?? []).length
        expect(colonCount).toBe(2)
    })

    it('test_exactly_60_incr_excludes_seconds', () => {
        // foundIncr === 60 is the boundary: seconds must NOT appear
        const result = formatWallClockTime(epoch, 60)
        expect(result).not.toContain('15')
    })

    it('test_exactly_60_incr_has_one_colon_separator', () => {
        const result = formatWallClockTime(epoch, 60)
        // h:mm produces one colon regardless of locale 12/24h choice
        const colonCount = (result.match(/:/g) ?? []).length
        expect(colonCount).toBe(1)
    })

    it('test_above_60_incr_excludes_seconds', () => {
        const result = formatWallClockTime(epoch, 300)
        expect(result).not.toContain('15')
        const colonCount = (result.match(/:/g) ?? []).length
        expect(colonCount).toBe(1)
    })

    it('test_minute_value_appears_in_output', () => {
        const result = formatWallClockTime(epoch, 60)
        expect(result).toContain('30')
    })
})
