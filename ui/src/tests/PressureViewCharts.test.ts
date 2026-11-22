import { describe, expect, it, vi, beforeEach } from 'vitest'
import { mount } from '@vue/test-utils'
import type uPlot from 'uplot'

const built: { opts: uPlot.Options; data: uPlot.AlignedData }[] = []

vi.mock('uplot', () => {
    function UPlotMock(
        this: Record<string, unknown>,
        opts: uPlot.Options,
        data: uPlot.AlignedData,
    ) {
        built.push({ opts, data })
        this.destroy = vi.fn()
        this.setSize = vi.fn()
        this.height = 240
    }
    return { default: UPlotMock }
})

vi.stubGlobal(
    'ResizeObserver',
    class {
        observe() {}
        disconnect() {}
    },
)

vi.mock('@/components/InfoHint.vue', () => ({
    default: { template: '<span class="info-hint-stub" />' },
}))

import RunningAhiChart from '@/components/RunningAhiChart.vue'
import TimeAtPressureChart from '@/components/TimeAtPressureChart.vue'
import type { RunningAhiSegment, TimeAtPressure } from '@/types'

function lastChart() {
    const chart = built.at(-1)
    if (!chart) throw new Error('no chart built')
    return chart
}

describe('RunningAhiChart', () => {
    beforeEach(() => {
        built.length = 0
    })

    it('test_segments_plot_per_minute_with_a_break_between', () => {
        const segments: RunningAhiSegment[] = [
            { session_id: 1, start: '2026-05-03T22:00:00', values: [0, 1] },
            { session_id: 1, start: '2026-05-03T23:30:00', values: [2] },
        ]
        mount(RunningAhiChart, { props: { segments } })

        const [xs, ys] = lastChart().data
        const start = new Date('2026-05-03T22:00:00').getTime() / 1000
        const second = new Date('2026-05-03T23:30:00').getTime() / 1000
        expect(xs).toEqual([start, start + 60, second - 30, second])
        expect(ys).toEqual([0, 1, null, 2])
    })

    it('test_no_segments_renders_nothing', () => {
        const wrapper = mount(RunningAhiChart, { props: { segments: [] } })

        expect(wrapper.find('section').exists()).toBe(false)
        expect(built).toHaveLength(0)
    })
})

describe('TimeAtPressureChart', () => {
    beforeEach(() => {
        built.length = 0
    })

    const tap: TimeAtPressure = {
        pressures: [8, 8.2],
        pressure_seconds: [120, 60],
        epap_seconds: null,
        event_counts: { OA: [0, 2] },
    }

    it('test_minutes_and_event_points_without_epap', () => {
        const wrapper = mount(TimeAtPressureChart, { props: { tap } })

        expect(wrapper.find('h2').text()).toContain('Time at Pressure')
        const { opts, data } = lastChart()
        expect(data).toEqual([
            [8, 8.2],
            [2, 1],
            [null, 2],
        ])
        expect(opts.series.map((s) => s.label)).toEqual([undefined, 'Pressure', 'OA'])
    })

    it('test_epap_series_appears_when_present', () => {
        mount(TimeAtPressureChart, { props: { tap: { ...tap, epap_seconds: [0, 30] } } })

        const { opts, data } = lastChart()
        expect(data[2]).toEqual([0, 0.5])
        expect(opts.series.map((s) => s.label)).toEqual([undefined, 'Pressure', 'EPAP', 'OA'])
    })

    it('test_null_renders_nothing', () => {
        const wrapper = mount(TimeAtPressureChart, { props: { tap: null } })

        expect(wrapper.find('section').exists()).toBe(false)
    })
})
