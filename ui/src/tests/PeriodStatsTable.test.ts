import { afterEach, describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import type { PeriodStatistics } from '@/types'
import PeriodStatsTable from '@/components/PeriodStatsTable.vue'
import { setMediaMatches } from './matchMedia'

const PERIOD = {
    period_start: '2026-09-01',
    period_end: '2026-09-30',
    days_used: 28,
    avg_hours_per_day: 7.1,
    avg_ahi: 2.3,
    median_ahi: 2.0,
    avg_pressure: 9.4,
    avg_leak: 3.8,
    avg_spo2: 95.1,
    avg_total_sleep_hours: 6.6,
    avg_sleep_efficiency_pct: 89.5,
} as unknown as PeriodStatistics

// Every column is a per-period aggregate, so every metric label carries exactly one Derived mark.
const METRIC_LABELS = [
    'Days Used',
    'Avg Hours',
    'Avg AHI',
    'Median AHI',
    'Avg Pressure',
    'Avg Leak',
    'Avg SpO₂',
    'Avg Sleep',
    'Avg Eff',
]

function derivedLabels(labels: Element[]): string[] {
    return labels
        .filter((el) => {
            const tiers = [...el.querySelectorAll('.provenance-mark')].map(
                (m) => m.getAttribute('aria-label')?.split(':')[0],
            )
            return tiers.length === 1 && tiers[0] === 'Derived'
        })
        .map((el) => el.textContent!.trim())
}

function mountTable() {
    return mount(PeriodStatsTable, {
        props: { periods: [PERIOD], loading: false, showSleepColumns: true },
    })
}

describe('PeriodStatsTable provenance', () => {
    afterEach(() => setMediaMatches(false))

    it('test_desktop_metric_headers_are_all_marked_derived', () => {
        const wrapper = mountTable()
        const headers = wrapper.findAll('th').map((th) => th.element)
        expect(derivedLabels(headers)).toEqual(METRIC_LABELS)
    })

    it('test_mobile_metric_labels_are_all_marked_derived', () => {
        setMediaMatches(true)
        const wrapper = mountTable()
        const labels = wrapper.findAll('.data-card-label').map((l) => l.element)
        expect(derivedLabels(labels)).toEqual(METRIC_LABELS)
    })
})
