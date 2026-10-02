import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import type { DayListItem } from '@/types'
import CalendarHeatmap from '@/components/CalendarHeatmap.vue'

// Same date math as the component (local clock, ISO slice) so the days land in the grid.
function isoDaysAgo(n: number): string {
    const d = new Date()
    d.setDate(d.getDate() - n)
    return d.toISOString().slice(0, 10)
}

function day(date: string, overrides: Partial<DayListItem> = {}): DayListItem {
    return { date, device_id: 1, session_count: 1, ...overrides }
}

function titleFor(days: DayListItem[], date: string): string | undefined {
    const wrapper = mount(CalendarHeatmap, { props: { days } })
    return wrapper
        .findAll('.cell')
        .map((c) => c.attributes('title'))
        .find((t) => t?.startsWith(`${date}:`))
}

describe('CalendarHeatmap cell titles', () => {
    it('test_device_index_source_names_value_device_reported', () => {
        const date = isoDaysAgo(2)
        const title = titleFor([day(date, { ahi: 3.24, index_source: 'device' })], date)
        expect(title).toBe(`${date}: AHI 3.2 (device-reported)`)
    })

    it('test_derived_index_source_names_value_snore_recount', () => {
        const date = isoDaysAgo(3)
        const title = titleFor([day(date, { ahi: 4.06, index_source: 'derived' })], date)
        expect(title).toBe(`${date}: AHI 4.1 (SNORE recount)`)
    })

    it('test_null_index_source_omits_source_and_missing_ahi_reads_na', () => {
        const withAhi = isoDaysAgo(4)
        const noAhi = isoDaysAgo(5)
        const days = [day(withAhi, { ahi: 1.5, index_source: null }), day(noAhi)]
        expect(titleFor(days, withAhi)).toBe(`${withAhi}: AHI 1.5`)
        expect(titleFor(days, noAhi)).toBe(`${noAhi}: AHI N/A`)
    })
})
