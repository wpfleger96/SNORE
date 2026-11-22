import { beforeEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'

// Hoisted so the uPlot mock factory can record what the component built.
const mockState = vi.hoisted(() => ({
    opts: null as Record<string, unknown> | null,
    data: null as unknown[][] | null,
}))

// Vitest requires a function or class body for constructor mocks.
vi.mock('uplot', () => {
    function UPlotMock(
        this: Record<string, unknown>,
        opts: Record<string, unknown>,
        data: unknown[][],
    ) {
        this.destroy = vi.fn()
        this.setSize = vi.fn()
        mockState.opts = opts
        mockState.data = data
    }
    ;(UPlotMock as unknown as Record<string, unknown>).paths = { bars: vi.fn(() => 'bars') }
    return { default: UPlotMock }
})

vi.mock('@/composables/useDarkMode', async () => {
    const { ref } = await vi.importActual<typeof import('vue')>('vue')
    return { useDarkMode: () => ({ isDark: ref(false) }) }
})

beforeEach(() => {
    mockState.opts = null
    mockState.data = null
    ;(globalThis as unknown as Record<string, unknown>).ResizeObserver = vi.fn(
        function ResizeObserverMock(this: Record<string, unknown>) {
            this.observe = vi.fn()
            this.disconnect = vi.fn()
        },
    )
})

import TrendChart from '@/components/TrendChart.vue'

type Series = { label: string; value?: (...a: unknown[]) => string; paths?: unknown }

const LABELS = ['2026-04-01', '2026-05-01', '2026-06-01']
const BREAKDOWN = [
    { label: 'OAI', values: [0.5, 0.4, null], color: '#a' },
    { label: 'CAI', values: [0.3, null, null], color: '#b' },
    { label: 'HI', values: [0.6, 0.5, null], color: '#c' },
]

function series(): Series[] {
    return (mockState.opts?.series as Series[]).slice(1)
}

describe('TrendChart', () => {
    it('test_stacked_bars_draw_running_totals_tallest_first', () => {
        mount(TrendChart, { props: { labels: LABELS, datasets: BREAKDOWN, stacked: true } })

        expect(series().map((s) => s.label)).toEqual(['HI', 'CAI', 'OAI'])
        expect(series().every((s) => s.paths === 'bars')).toBe(true)
        const [, hi, cai, oai] = mockState.data!
        // A missing component adds nothing; a period with no component has no bar.
        expect(hi).toEqual([1.4, 0.9, undefined])
        expect(cai).toEqual([0.8, 0.4, undefined])
        expect(oai).toEqual([0.5, 0.4, undefined])
    })

    it('test_stacked_legend_shows_each_component_not_its_running_total', () => {
        mount(TrendChart, { props: { labels: LABELS, datasets: BREAKDOWN, stacked: true } })

        const [hi, cai] = series()
        expect(hi.value!(null, 1.4, 1, 0)).toBe('0.6')
        expect(cai.value!(null, 0.4, 2, 1)).toBe('--')
    })

    it('test_format_value_formats_axis_ticks_and_legend', () => {
        const format = (v: number) => `${v}h`
        mount(TrendChart, {
            props: {
                labels: LABELS.slice(0, 1),
                datasets: [{ label: 'Bedtime', values: [11.5], color: '#a' }],
                formatValue: format,
            },
        })

        const yAxis = (mockState.opts?.axes as { values?: (...a: unknown[]) => string[] }[])[1]
        expect(yAxis.values!(null, [11, 12])).toEqual(['11h', '12h'])
        expect(series()[0].value!(null, 11.5, 1, 0)).toBe('11.5h')
    })

    it('test_plain_chart_keeps_uplot_default_legend_and_ticks', () => {
        mount(TrendChart, {
            props: {
                labels: LABELS.slice(0, 1),
                datasets: [{ label: 'AHI', values: [1.2], color: '#a' }],
            },
        })

        expect(series()[0].value).toBeUndefined()
        expect((mockState.opts?.axes as { values?: unknown }[])[1].values).toBeUndefined()
    })
})
