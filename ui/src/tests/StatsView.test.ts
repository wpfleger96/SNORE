import { afterEach, describe, expect, it, vi, beforeEach } from 'vitest'
import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import { defineComponent } from 'vue'

// uPlot is instantiated at module scope in StatsView; mock it before the view is imported.
vi.mock('uplot', () => ({
    default: { sync: vi.fn(() => ({ plots: [] })) },
}))

vi.mock('@/composables/useAuth')

vi.mock('@/api/stats', () => ({
    getPeriods: vi.fn(),
    getTrends: vi.fn(),
    getRecords: vi.fn(),
    getDataRange: vi.fn(),
}))

// Return input unchanged so tests can assert on the raw ISO date string.
vi.mock('@/utils/formatting', () => ({
    formatDateFull: (d: string) => d,
    formatDateMonthDay: (d: string) => d,
    ahiClass: () => '',
}))

vi.mock('@/components/ui/toggle-group', () => ({
    ToggleGroup: defineComponent({
        name: 'ToggleGroup',
        props: ['modelValue', 'type', 'variant'],
        emits: ['update:model-value'],
        template: '<div><slot /></div>',
    }),
    ToggleGroupItem: {
        props: ['value', 'disabled'],
        template: '<button :data-value="value"><slot /></button>',
    },
}))

vi.mock('@/components/TrendChart.vue', () => ({
    default: defineComponent({
        name: 'TrendChart',
        props: ['labels', 'datasets', 'height', 'syncKey'],
        template: '<canvas class="trend-chart-stub" />',
    }),
}))

// Expose emptyMessage as a data attribute so tests can assert on it.
vi.mock('@/components/PeriodStatsTable.vue', () => ({
    default: defineComponent({
        name: 'PeriodStatsTable',
        props: {
            periods: { type: Array, default: () => [] },
            loading: { type: Boolean, default: false },
            emptyMessage: { type: String, default: undefined },
        },
        template: `<div class="period-stats-table-stub" :data-empty-message="emptyMessage" />`,
    }),
}))

vi.mock('@/components/RecordsPanel.vue', () => ({
    default: defineComponent({
        name: 'RecordsPanel',
        props: {
            records: { default: null },
            loading: { type: Boolean, default: false },
            emptyMessage: { type: String, default: undefined },
        },
        template: `<div class="records-panel-stub" :data-empty-message="emptyMessage" />`,
    }),
}))

vi.mock('@/components/ErrorState.vue', () => ({
    default: { template: '<div class="error-state-stub" />' },
}))

import { makeAuthMock } from './helpers/mockUseAuth'
import { useAuth } from '@/composables/useAuth'
import { getPeriods, getTrends, getRecords, getDataRange } from '@/api/stats'
import type { PeriodStatistics } from '@/types'
import StatsView from '@/views/StatsView.vue'

const EMPTY_TRENDS = { ahi: [], usage: [], spo2: [], leak: [] }

function setupAuthMock() {
    vi.mocked(useAuth).mockReturnValue(makeAuthMock() as never)
}

async function mountAndLoad() {
    const wrapper = mount(StatsView)
    await flushPromises()
    return wrapper
}

describe('StatsView', () => {
    beforeEach(() => {
        vi.clearAllMocks()
        setupAuthMock()
        vi.mocked(getPeriods).mockResolvedValue([])
        vi.mocked(getTrends).mockResolvedValue(EMPTY_TRENDS)
        vi.mocked(getRecords).mockResolvedValue({})
        vi.mocked(getDataRange).mockResolvedValue({ latest_date: null })
    })

    it('test_periods_and_trends_called_with_numeric_days_limit', async () => {
        // Default daysRange is '90d' which maps to 90 in rangeDaysMap.
        await mountAndLoad()

        expect(getPeriods).toHaveBeenCalledWith('month', 90)
        expect(getTrends).toHaveBeenCalledWith('month', 90)
    })

    it('test_periods_empty_message_includes_date_and_hint_when_data_range_known', async () => {
        vi.mocked(getDataRange).mockResolvedValue({ latest_date: '2026-05-03' })
        vi.mocked(getPeriods).mockResolvedValue([])

        const wrapper = await mountAndLoad()

        const table = wrapper.find('.period-stats-table-stub')
        const msg = table.attributes('data-empty-message') ?? ''
        // The date formatter is mocked to return the ISO string as-is.
        expect(msg).toContain('2026-05-03')
        expect(msg).toContain('Try a wider range')
    })

    it('test_records_empty_message_includes_date_and_hint_when_data_range_known', async () => {
        vi.mocked(getDataRange).mockResolvedValue({ latest_date: '2026-05-03' })
        vi.mocked(getRecords).mockResolvedValue({})

        const wrapper = await mountAndLoad()

        const panel = wrapper.find('.records-panel-stub')
        const msg = panel.attributes('data-empty-message') ?? ''
        // The date formatter is mocked to return the ISO string as-is.
        expect(msg).toContain('2026-05-03')
        expect(msg).toContain('Try a wider range')
    })

    it('test_periods_and_trends_called_with_undefined_when_range_is_all', async () => {
        const wrapper = await mountAndLoad()

        // The Range toggle is the second ToggleGroup (after Granularity, before Metrics).
        const rangeToggle = wrapper.findAllComponents({ name: 'ToggleGroup' })[1]
        await rangeToggle.vm.$emit('update:model-value', 'all')
        await flushPromises()

        expect(getPeriods).toHaveBeenLastCalledWith('month', undefined)
        expect(getTrends).toHaveBeenLastCalledWith('month', undefined)
    })

    describe('RERA trend', () => {
        const TRENDS = { ...EMPTY_TRENDS, ahi: [['2026-04-01', 1.2]] as [string, number | null][] }
        const reraPoints = (value: number | null) =>
            [
                ['2026-04-01', value],
                ['2026-05-01', value],
            ] as [string, number | null][]

        beforeEach(() => {
            localStorage.setItem('snore:trend-metrics', JSON.stringify(['rera']))
        })

        afterEach(() => {
            localStorage.clear()
        })

        it('test_all_null_series_offers_no_toggle_or_chart', async () => {
            vi.mocked(getTrends).mockResolvedValue({ ...TRENDS, rera: reraPoints(null) })

            const wrapper = await mountAndLoad()

            expect(wrapper.find('[data-value="rera"]').exists()).toBe(false)
            expect(wrapper.find('[data-value="ahi"]').exists()).toBe(true)
            expect(wrapper.find('.trend-chart-stub').exists()).toBe(false)
        })

        it('test_series_of_real_zeros_keeps_toggle_and_chart', async () => {
            vi.mocked(getTrends).mockResolvedValue({ ...TRENDS, rera: reraPoints(0) })

            const wrapper = await mountAndLoad()

            expect(wrapper.find('[data-value="rera"]').exists()).toBe(true)
            expect(wrapper.find('.trend-chart-stub').exists()).toBe(true)
        })

        it('test_selected_rera_returns_when_a_wider_range_has_data', async () => {
            vi.mocked(getTrends).mockResolvedValue({ ...TRENDS, rera: reraPoints(null) })
            const wrapper = await mountAndLoad()
            vi.mocked(getTrends).mockResolvedValue({ ...TRENDS, rera: reraPoints(0.4) })

            const rangeToggle = wrapper.findAllComponents({ name: 'ToggleGroup' })[1]
            await rangeToggle.vm.$emit('update:model-value', 'all')
            await flushPromises()

            expect(wrapper.find('[data-value="rera"]').exists()).toBe(true)
            expect(wrapper.find('.trend-chart-stub').exists()).toBe(true)
        })
    })

    describe('device channel trends', () => {
        const LABELS = { ahi: [['2026-04-01', 1.2]] as [string, number | null][] }
        const series = (value: number | null) =>
            [['2026-04-01', value]] as [string, number | null][]
        const period = (start: string, flg: number | null, snore: number | null) =>
            ({
                period_type: 'month',
                period_start: start,
                period_end: start,
                device_flg_95th: flg,
                snore_95th: snore,
            }) as PeriodStatistics

        afterEach(() => {
            localStorage.clear()
        })

        it.each(['device_flg_95th', 'snore_95th'])(
            'test_all_null_%s_offers_no_toggle',
            async (key) => {
                vi.mocked(getTrends).mockResolvedValue({ ...EMPTY_TRENDS, ...LABELS })
                vi.mocked(getPeriods).mockResolvedValue([period('2026-04-01', null, null)])

                const wrapper = await mountAndLoad()

                expect(wrapper.find(`[data-value="${key}"]`).exists()).toBe(false)
            },
        )

        it.each(['device_flg_95th', 'snore_95th'])(
            'test_series_of_real_zeros_keeps_%s_toggle_and_chart',
            async (key) => {
                localStorage.setItem('snore:trend-metrics', JSON.stringify([key]))
                vi.mocked(getTrends).mockResolvedValue({ ...EMPTY_TRENDS, ...LABELS })
                vi.mocked(getPeriods).mockResolvedValue([period('2026-04-01', 0, 0)])

                const wrapper = await mountAndLoad()

                expect(wrapper.find(`[data-value="${key}"]`).exists()).toBe(true)
                expect(wrapper.find('.trend-chart-stub').exists()).toBe(true)
            },
        )

        it('test_chart_values_align_to_trend_labels_by_period_start', async () => {
            localStorage.setItem('snore:trend-metrics', JSON.stringify(['device_flg_95th']))
            vi.mocked(getTrends).mockResolvedValue({
                ...EMPTY_TRENDS,
                ahi: [
                    ['2026-02-01', 1],
                    ['2026-03-01', 1],
                    ['2026-04-01', 1],
                ],
            })
            // Out of order, and 2026-03 is missing.
            vi.mocked(getPeriods).mockResolvedValue([
                period('2026-04-01', 0.4, 0.1),
                period('2026-02-01', 0.2, 0.3),
            ])

            const wrapper = await mountAndLoad()

            const chart = wrapper.findComponent({ name: 'TrendChart' })
            expect(chart.props('datasets')[0].values).toEqual([0.2, null, 0.4])
        })

        it('test_always_on_series_keep_their_toggle_when_all_null', async () => {
            vi.mocked(getTrends).mockResolvedValue({
                ...EMPTY_TRENDS,
                ...LABELS,
                spo2: series(null),
            })

            const wrapper = await mountAndLoad()

            expect(wrapper.find('[data-value="spo2"]').exists()).toBe(true)
        })
    })

    describe('stored trend selection', () => {
        const selectedMetrics = (wrapper: VueWrapper) =>
            wrapper.findAllComponents({ name: 'ToggleGroup' })[2].props('modelValue')

        afterEach(() => {
            localStorage.clear()
        })

        it('test_selection_saved_before_the_device_series_existed_still_loads', async () => {
            localStorage.setItem('snore:trend-metrics', JSON.stringify(['ahi', 'pressure', 'rera']))

            const wrapper = await mountAndLoad()

            expect(selectedMetrics(wrapper)).toEqual(['ahi', 'pressure', 'rera'])
        })

        it('test_saved_device_series_survive_a_reload_and_retired_keys_drop', async () => {
            localStorage.setItem(
                'snore:trend-metrics',
                JSON.stringify(['device_flg_95th', 'snore_95th', 'retired_metric']),
            )

            const wrapper = await mountAndLoad()

            expect(selectedMetrics(wrapper)).toEqual(['device_flg_95th', 'snore_95th'])
        })

        it('test_no_stored_selection_falls_back_to_defaults_without_device_series', async () => {
            const wrapper = await mountAndLoad()

            expect(selectedMetrics(wrapper)).toEqual(['ahi', 'usage', 'spo2', 'leak'])
        })
    })
})
