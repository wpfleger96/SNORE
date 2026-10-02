// @vitest-environment node
import { describe, expect, it } from 'vitest'
import {
    getByPath,
    computeDelta,
    diffKeys,
    identityChips,
    formatMetric,
    formatDelta,
    AGGREGATE_METRICS,
    type MetricDescriptor,
} from '@/utils/validationMetrics'

describe('getByPath', () => {
    it('test_reads_nested_numeric_by_dotted_path', () => {
        expect(getByPath({ rr: { mean_spearman_r: 0.42 } }, 'rr.mean_spearman_r')).toBe(0.42)
    })

    it('test_missing_path_returns_null', () => {
        expect(getByPath({ rr: {} }, 'rr.mean_spearman_r')).toBeNull()
        expect(getByPath(null, 'a.b')).toBeNull()
    })

    it('test_non_finite_or_non_numeric_returns_null', () => {
        expect(getByPath({ a: NaN }, 'a')).toBeNull()
        expect(getByPath({ a: 'x' }, 'a')).toBeNull()
    })
})

describe('computeDelta', () => {
    const metric: MetricDescriptor = {
        path: 'mean_proxy_sensitivity',
        label: 'Proxy sensitivity',
        kind: 'percent',
        higherIsBetter: true,
    }

    it('test_improvement_when_higher_is_better_and_b_greater', () => {
        const d = computeDelta(
            { mean_proxy_sensitivity: 0.1 },
            { mean_proxy_sensitivity: 0.25 },
            metric,
        )
        expect(d.delta).toBeCloseTo(0.15)
        expect(d.direction).toBe('better')
    })

    it('test_regression_when_higher_is_better_and_b_smaller', () => {
        const d = computeDelta(
            { mean_proxy_sensitivity: 0.3 },
            { mean_proxy_sensitivity: 0.2 },
            metric,
        )
        expect(d.direction).toBe('worse')
    })

    it('test_lower_is_better_flips_direction', () => {
        const errMetric: MetricDescriptor = {
            path: 'mae',
            label: 'MAE',
            kind: 'rate',
            higherIsBetter: false,
        }
        expect(computeDelta({ mae: 5 }, { mae: 3 }, errMetric).direction).toBe('better')
        expect(computeDelta({ mae: 3 }, { mae: 5 }, errMetric).direction).toBe('worse')
    })

    it('test_null_side_yields_null_delta_and_neutral', () => {
        const d = computeDelta({}, { mean_proxy_sensitivity: 0.2 }, metric)
        expect(d.a).toBeNull()
        expect(d.delta).toBeNull()
        expect(d.direction).toBe('neutral')
    })

    it('test_zero_delta_is_neutral', () => {
        const d = computeDelta(
            { mean_proxy_sensitivity: 0.2 },
            { mean_proxy_sensitivity: 0.2 },
            metric,
        )
        expect(d.delta).toBe(0)
        expect(d.direction).toBe('neutral')
    })

    it('test_count_metric_without_direction_stays_neutral', () => {
        const countMetric: MetricDescriptor = { path: 'n', label: 'N', kind: 'count' }
        expect(computeDelta({ n: 1 }, { n: 9 }, countMetric).direction).toBe('neutral')
    })
})

describe('diffKeys', () => {
    it('test_identifies_changed_and_added_keys', () => {
        const a = { fl_classifier: 'v1', recovery_detector: 'v2' }
        const b = { fl_classifier: 'v2', recovery_detector: 'v2', extra: 'x' }
        const diff = diffKeys(a, b)
        expect(diff.has('fl_classifier')).toBe(true)
        expect(diff.has('recovery_detector')).toBe(false)
        expect(diff.has('extra')).toBe(true)
    })

    it('test_identical_objects_have_empty_diff', () => {
        expect(diffKeys({ a: 1, b: 2 }, { a: 1, b: 2 }).size).toBe(0)
    })

    it('test_numeric_vs_string_values_differ', () => {
        expect(diffKeys({ a: 2 }, { a: '2' }).has('a')).toBe(true)
    })

    it('test_null_inputs_treated_as_empty', () => {
        expect(diffKeys(null, { a: 1 }).has('a')).toBe(true)
        expect(diffKeys(undefined, undefined).size).toBe(0)
    })
})

describe('identityChips', () => {
    it('test_flattens_sorted_key_value_chips', () => {
        expect(identityChips({ recovery_detector: 'v2', fl_classifier: 'v1' })).toEqual([
            'fl_classifier v1',
            'recovery_detector v2',
        ])
    })

    it('test_nullish_yields_empty', () => {
        expect(identityChips(null)).toEqual([])
        expect(identityChips(undefined)).toEqual([])
    })
})

describe('formatMetric / formatDelta', () => {
    it('test_percent_formats_ratio_and_signed_pp_delta', () => {
        expect(formatMetric(0.25, 'percent')).toBe('25.0%')
        expect(formatDelta(0.05, 'percent')).toBe('+5.0 pp')
        expect(formatDelta(-0.05, 'percent')).toBe('-5.0 pp')
    })

    it('test_decimal_and_count_and_null', () => {
        expect(formatMetric(0.123456, 'decimal')).toBe('0.123')
        expect(formatMetric(7, 'count')).toBe('7')
        expect(formatMetric(null, 'decimal')).toBe('—')
        expect(formatDelta(null, 'decimal')).toBe('—')
        expect(formatDelta(2, 'count')).toBe('+2')
    })

    it('test_chance_floor_scale_percent_stays_visible', () => {
        // The RERA comparison must represent movement at the chance-floor scale
        // (~4e-5) instead of flooring both value and delta to 0.
        expect(formatMetric(4e-5, 'percent')).toBe('0.0040%')
        expect(formatDelta(4e-5, 'percent')).toBe('+0.0040 pp')
        expect(formatDelta(-4e-5, 'percent')).toBe('-0.0040 pp')
    })
})

describe('AGGREGATE_METRICS coverage', () => {
    it('test_every_validator_type_has_metrics', () => {
        for (const type of ['events', 'fl', 'breaths', 'rera', 'apple'] as const) {
            expect(AGGREGATE_METRICS[type].length).toBeGreaterThan(0)
        }
    })
})
