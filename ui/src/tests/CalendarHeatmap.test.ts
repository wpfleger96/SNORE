import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
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

function monthName(year: number, monthIndex: number): string {
    return new Date(year, monthIndex, 15).toLocaleString(undefined, { month: 'short' })
}

function renderedMonthLabels(): { text: string; column: number }[] {
    const wrapper = mount(CalendarHeatmap, { props: { days: [] } })
    return wrapper.findAll('.month-labels span').map((s) => ({
        text: s.text(),
        column: Number(/grid-column:\s*(\d+)/.exec(s.attributes('style') ?? '')?.[1]),
    }))
}

const APR_TO_SEP_2026 = [3, 4, 5, 6, 7, 8].map((m) => monthName(2026, m))
const APR_TO_OCT_2026 = [3, 4, 5, 6, 7, 8, 9].map((m) => monthName(2026, m))

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

describe('CalendarHeatmap month labels', () => {
    beforeEach(() => vi.useFakeTimers())
    afterEach(() => {
        vi.useRealTimers()
    })

    it('test_monday_spillover_into_prior_month_drops_spillover_label', () => {
        // 6 months back is Fri 2026-04-03; Monday alignment starts the grid on Mar 30.
        vi.setSystemTime(new Date(2026, 9, 3, 12))

        const labels = renderedMonthLabels()

        expect(labels[0].text).toBe(monthName(2026, 3))
        for (let i = 1; i < labels.length; i++) {
            expect(labels[i].column - labels[i - 1].column).toBeGreaterThanOrEqual(2)
        }
        // The last week column starts Mon Sep 28, so no column belongs to October yet.
        expect(labels.map((l) => l.text)).toEqual(APR_TO_SEP_2026)
    })

    it('test_mid_month_start_labels_every_month_once', () => {
        // 6 months back is Wed 2026-04-15; Monday alignment stays in April.
        vi.setSystemTime(new Date(2026, 9, 15, 12))

        const labels = renderedMonthLabels()

        expect(labels.map((l) => l.text)).toEqual(APR_TO_OCT_2026)
        expect(labels[0].column).toBe(1)
    })

    it('test_late_month_start_keeps_two_column_first_month', () => {
        // 6 months back is Sat Apr 25 → grid starts Mon Apr 20; Apr covers
        // columns 1–2, May starts column 3.
        vi.setSystemTime(new Date(2026, 9, 25, 12))

        const labels = renderedMonthLabels()

        expect(labels[0].text).toBe(monthName(2026, 3))
        expect(labels[0].column).toBe(1)
        expect(labels[1].text).toBe(monthName(2026, 4))
        expect(labels[1].column).toBe(3)
    })

    it('test_month_labels_share_column_with_grid_above_cells', () => {
        // Guards against the side-by-side layout regression: .month-labels and
        // .grid must share a wrapper div that is not the .calendar-heatmap root.
        const wrapper = mount(CalendarHeatmap, { props: { days: [] } })
        const monthLabelsEl = wrapper.find('.month-labels').element
        const gridEl = wrapper.find('.grid').element
        const calendarHeatmapEl = wrapper.find('.calendar-heatmap').element

        expect(monthLabelsEl.parentElement).toBe(gridEl.parentElement)
        expect(monthLabelsEl.parentElement).not.toBe(calendarHeatmapEl)
        expect(monthLabelsEl.nextElementSibling).toBe(gridEl)
    })
})
