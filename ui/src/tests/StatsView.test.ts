import { describe, expect, it, vi, beforeEach } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { defineComponent } from 'vue'

// uPlot is instantiated at module scope in StatsView; mock it before the view is imported.
vi.mock('uplot', () => ({
    default: { sync: vi.fn(() => ({ plots: [] })) },
}))

vi.mock('@/composables/useAuth')

vi.mock('@/api/stats', () => ({
    getSummary: vi.fn(),
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
        template: '<button><slot /></button>',
    },
}))

vi.mock('@/components/TrendChart.vue', () => ({
    default: { template: '<canvas class="trend-chart-stub" />' },
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
import { getSummary, getPeriods, getTrends, getRecords, getDataRange } from '@/api/stats'
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
        vi.mocked(getSummary).mockResolvedValue(null)
        vi.mocked(getPeriods).mockResolvedValue([])
        vi.mocked(getTrends).mockResolvedValue(EMPTY_TRENDS)
        vi.mocked(getRecords).mockResolvedValue({})
        vi.mocked(getDataRange).mockResolvedValue({ latest_date: null })
    })

    it('test_badge_hidden_when_summary_is_null', async () => {
        vi.mocked(getSummary).mockResolvedValue(null)

        const wrapper = await mountAndLoad()

        expect(wrapper.find('.trend-badge').exists()).toBe(false)
    })

    it('test_badge_shown_when_ahi_trend_is_worsening', async () => {
        vi.mocked(getSummary).mockResolvedValue({
            ahi_trend_direction: 'worsening',
        } as never)

        const wrapper = await mountAndLoad()

        const badge = wrapper.find('.trend-badge')
        expect(badge.exists()).toBe(true)
        expect(badge.text()).toContain('worsening')
    })

    it('test_getSummary_called_with_numeric_days_limit', async () => {
        // Default daysRange is '90d' which maps to 90 in rangeDaysMap.
        await mountAndLoad()

        expect(getSummary).toHaveBeenCalledWith(90)
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

    it('test_getSummary_called_with_undefined_when_range_is_all', async () => {
        const wrapper = await mountAndLoad()

        // The Range toggle is the second ToggleGroup (after Granularity, before Metrics).
        const rangeToggle = wrapper.findAllComponents({ name: 'ToggleGroup' })[1]
        await rangeToggle.vm.$emit('update:model-value', 'all')
        await flushPromises()

        expect(getSummary).toHaveBeenLastCalledWith(undefined)
    })
})
