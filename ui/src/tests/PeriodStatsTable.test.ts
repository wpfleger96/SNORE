import { afterEach, describe, expect, it, vi } from 'vitest'
import { mount, type VueWrapper } from '@vue/test-utils'
import { nextTick } from 'vue'
import type { PeriodStatistics } from '@/types'
import { setMediaMatches } from './matchMedia'

vi.mock('@/components/InfoHint.vue', () => ({
    default: { template: '<span class="info-hint-stub" />' },
}))

import PeriodStatsTable from '@/components/PeriodStatsTable.vue'

function makePeriod(overrides: Partial<PeriodStatistics> = {}): PeriodStatistics {
    return {
        period_type: 'month',
        period_start: '2026-04-01',
        period_end: '2026-04-30',
        days_used: 6,
        days_in_period: 30,
        ...overrides,
    } as PeriodStatistics
}

function mountPeriods(periods: PeriodStatistics[]): VueWrapper {
    return mount(PeriodStatsTable, { props: { periods, loading: false } })
}

function headers(wrapper: VueWrapper): string[] {
    return wrapper.findAll('th').map((th) => th.text())
}

/** The body cell under a header; undefined when that column is hidden. */
function columnCell(wrapper: VueWrapper, header: string, rowIndex = 0): string | undefined {
    const index = headers(wrapper).indexOf(header)
    if (index < 0) return undefined
    return wrapper.findAll('tbody tr')[rowIndex].findAll('td')[index].text()
}

describe('PeriodStatsTable device channel columns', () => {
    afterEach(() => {
        setMediaMatches(false)
    })

    it('test_all_null_periods_hide_both_columns', () => {
        const wrapper = mountPeriods([
            makePeriod({ device_flg_95th: null, snore_95th: null }),
            makePeriod({ period_start: '2026-05-01' }),
        ])

        expect(headers(wrapper)).not.toContain('FL 95th')
        expect(headers(wrapper)).not.toContain('Snore 95th')
    })

    it('test_columns_are_independent', () => {
        const wrapper = mountPeriods([makePeriod({ device_flg_95th: 0.21, snore_95th: null })])

        expect(columnCell(wrapper, 'FL 95th')).toBe('0.21')
        expect(headers(wrapper)).not.toContain('Snore 95th')
    })

    it('test_one_period_with_a_value_shows_the_column_with_a_placeholder_elsewhere', () => {
        const wrapper = mountPeriods([
            makePeriod({ device_flg_95th: null }),
            makePeriod({ period_start: '2026-05-01', device_flg_95th: 0.3 }),
        ])

        expect(columnCell(wrapper, 'FL 95th', 0)).toBe('---')
        expect(columnCell(wrapper, 'FL 95th', 1)).toBe('0.30')
    })

    it('test_zero_counts_as_a_value', () => {
        const wrapper = mountPeriods([makePeriod({ device_flg_95th: 0, snore_95th: 0 })])

        expect(columnCell(wrapper, 'FL 95th')).toBe('0.00')
        expect(columnCell(wrapper, 'Snore 95th')).toBe('0.00')
    })

    it('test_mobile_cards_follow_the_same_rule', async () => {
        setMediaMatches(true)
        const wrapper = mountPeriods([makePeriod({ device_flg_95th: 0, snore_95th: null })])
        await nextTick()

        const rows = wrapper.findAll('.data-card-row')
        const flRow = rows.find((r) => r.find('.data-card-label').text() === 'FL 95th')
        expect(flRow?.find('.data-card-value').text()).toBe('0.00')
        expect(rows.some((r) => r.find('.data-card-label').text() === 'Snore 95th')).toBe(false)
    })
})
