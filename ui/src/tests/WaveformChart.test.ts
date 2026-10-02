import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { mount } from '@vue/test-utils'
import { ZOOM_FETCH_DEBOUNCE_MS } from '@/constants/waveform'

// Shared handle so tests can reach the uPlot instance the component created.
// Hoisted because vi.mock factories run before module-level declarations.
const mockState = vi.hoisted(() => ({
    lastChart: null as Record<string, unknown> | null,
    lastOpts: null as Record<string, unknown> | null,
}))

// uPlot must be mocked before the component is imported.
// Vitest requires a 'function' or 'class' body for constructor mocks.
//
// The mock mirrors the two pieces of real uPlot behavior the floor logic rides
// on: (1) setScale runs the x scale's `range` fn to derive the committed extent,
// then (2) fires the setScale hooks. Construction performs the same initial
// autoscale + hook fire real uPlot does, which flips the component's
// isInitialRender flag exactly as production does.
vi.mock('uplot', () => {
    function UPlotMock(
        this: Record<string, unknown>,
        opts: Record<string, unknown>,
        data: number[][],
    ) {
        const scalesOpt = (opts.scales ?? {}) as Record<
            string,
            { range?: (...a: unknown[]) => [number, number] }
        >
        const hooks = (opts.hooks ?? {}) as { setScale?: Array<(u: unknown, key: string) => void> }

        const applyRange = (key: string, min: number, max: number): [number, number] => {
            const range = scalesOpt[key]?.range
            return range ? range(null, min, max, key) : [min, max]
        }

        this.scales = {} as Record<string, { min: number; max: number }>
        this.destroy = vi.fn()
        this.setData = vi.fn()
        this.setSize = vi.fn()
        this.setScale = vi.fn(function (
            this: Record<string, unknown>,
            key: string,
            range: { min: number; max: number },
        ) {
            const [min, max] = applyRange(key, range.min, range.max)
            ;(this.scales as Record<string, { min: number; max: number }>)[key] = { min, max }
            for (const hook of hooks.setScale ?? []) hook(this, key)
        })
        this.batch = vi.fn()
        this.over = document.createElement('div')
        this.height = 240
        this.redraw = vi.fn()
        this.data = data

        // Initial autoscale + setScale fire, as real uPlot does on construction.
        const xs = data?.[0]
        if (xs?.length) {
            const [min, max] = applyRange('x', xs[0], xs[xs.length - 1])
            ;(this.scales as Record<string, { min: number; max: number }>).x = { min, max }
            for (const hook of hooks.setScale ?? []) hook(this, 'x')
        }

        mockState.lastChart = this
        mockState.lastOpts = opts
    }
    ;(UPlotMock as unknown as Record<string, unknown>).sync = vi.fn(() => ({
        key: 'k',
        sub: vi.fn(),
        unsub: vi.fn(),
    }))
    ;(UPlotMock as unknown as Record<string, unknown>).pxRatio = 1
    return { default: UPlotMock }
})

// Stub lucide icons used inside WaveformChart.
vi.mock('@lucide/vue', () => ({
    Loader2: { template: '<svg class="loader2-stub animate-spin" />' },
}))

// useDarkMode must return a real Vue ref so the watch() inside WaveformChart works.
vi.mock('@/composables/useDarkMode', async () => {
    const { ref } = await vi.importActual<typeof import('vue')>('vue')
    return {
        useDarkMode: () => ({ isDark: ref(false) }),
    }
})

// Stub formatting utility.
vi.mock('@/utils/formatting', () => ({
    formatWallClockTime: (v: number) => String(v),
}))

// Stub ResizeObserver (jsdom does not provide it).
// Vitest requires a function or class body for constructor mocks.
beforeEach(() => {
    mockState.lastChart = null
    mockState.lastOpts = null
    ;(globalThis as unknown as Record<string, unknown>).ResizeObserver = vi.fn(
        function ResizeObserverMock(this: Record<string, unknown>) {
            this.observe = vi.fn()
            this.unobserve = vi.fn()
            this.disconnect = vi.fn()
        },
    )
})

import WaveformChart from '@/components/WaveformChart.vue'

const MINIMAL_PROPS = {
    timestamps: [0, 1],
    values: [0, 1],
    unit: 'L/min',
    label: 'Flow',
    startEpoch: 0,
}

// A session long enough that the 5 s floor bites well inside the data span.
const LONG_PROPS = {
    timestamps: [0, 100],
    values: [0, 1],
    unit: 'L/min',
    label: 'Flow',
    startEpoch: 0,
}

function chartOf(): {
    scales: { x?: { min: number; max: number } }
    setScale: (key: string, range: { min: number; max: number }) => void
} {
    const chart = mockState.lastChart
    if (!chart) throw new Error('uPlot chart was not created')
    return chart as unknown as {
        scales: { x?: { min: number; max: number } }
        setScale: (key: string, range: { min: number; max: number }) => void
    }
}

describe('WaveformChart', () => {
    it('test_sanity_mounts_without_errors_given_minimal_props', () => {
        expect(() => mount(WaveformChart, { props: MINIMAL_PROPS })).not.toThrow()
    })

    it('test_event_overlay_is_labelled_device_scored', () => {
        const event = {
            id: 1,
            event_type: 'OA',
            start_time: 0,
            duration_seconds: 10,
            offset_seconds: 0,
            spo2_drop: null,
            peak_flow_limitation: null,
        }
        const withEvents = mount(WaveformChart, { props: { ...MINIMAL_PROPS, events: [event] } })
        expect(withEvents.text()).toContain('Device-scored events')

        const without = mount(WaveformChart, { props: MINIMAL_PROPS })
        expect(without.text()).not.toContain('Device-scored events')
    })

    it('test_refetching_true_renders_corner_spinner', () => {
        const wrapper = mount(WaveformChart, {
            props: { ...MINIMAL_PROPS, refetching: true },
        })

        expect(wrapper.find('.animate-spin').exists()).toBe(true)
    })

    it('test_refetching_false_hides_corner_spinner', () => {
        const wrapper = mount(WaveformChart, {
            props: { ...MINIMAL_PROPS, refetching: false },
        })

        expect(wrapper.find('.animate-spin').exists()).toBe(false)
    })

    it('test_refetching_omitted_hides_corner_spinner', () => {
        const wrapper = mount(WaveformChart, {
            props: MINIMAL_PROPS,
        })

        expect(wrapper.find('.animate-spin').exists()).toBe(false)
    })

    it('test_sub_floor_programmatic_zoom_commits_five_second_span', () => {
        const wrapper = mount(WaveformChart, { props: LONG_PROPS })
        // Programmatic zoom to a 1 s window; the range fn floors it to 5 s.
        ;(wrapper.vm as unknown as { setScaleX: (min: number, max: number) => void }).setScaleX(
            10,
            11,
        )

        const x = chartOf().scales.x!
        expect(x.max - x.min).toBeCloseTo(5, 10)
        // Floor expands around the window center (10.5), staying inside the data.
        expect(x.min).toBeCloseTo(8, 10)
        expect(x.max).toBeCloseTo(13, 10)
    })

    it('test_sub_floor_drag_zoom_commits_five_second_span', () => {
        mount(WaveformChart, { props: LONG_PROPS })
        // Simulate a native uPlot drag-zoom (component's setScaleX is NOT involved).
        chartOf().setScale('x', { min: 10, max: 11 })

        const x = chartOf().scales.x!
        expect(x.max - x.min).toBeCloseTo(5, 10)
    })

    it('test_above_floor_zoom_passes_through_unclamped', () => {
        mount(WaveformChart, { props: LONG_PROPS })
        chartOf().setScale('x', { min: 10, max: 30 })

        const x = chartOf().scales.x!
        expect(x.min).toBeCloseTo(10, 10)
        expect(x.max).toBeCloseTo(30, 10)
    })

    it('test_short_session_floor_equals_data_span', () => {
        mount(WaveformChart, {
            props: { ...LONG_PROPS, timestamps: [0, 3], values: [0, 1] },
        })
        // Data span is 3 s (< 5 s floor); any sub-span floors to the full 3 s.
        chartOf().setScale('x', { min: 1, max: 1.5 })

        const x = chartOf().scales.x!
        expect(x.min).toBeCloseTo(0, 10)
        expect(x.max).toBeCloseTo(3, 10)
    })

    it('test_data_series_disables_point_markers', () => {
        mount(WaveformChart, { props: LONG_PROPS })

        const opts = mockState.lastOpts as unknown as {
            series: Array<{ points?: { show?: boolean } }>
        }
        // Dot-flash regression guard: point markers stay hidden even when a slice is sparse.
        expect(opts.series[1].points!.show).toBe(false)
    })

    it('test_data_update_swaps_without_rescaling_the_viewport', async () => {
        const wrapper = mount(WaveformChart, { props: LONG_PROPS })
        const chart = mockState.lastChart as unknown as { setData: ReturnType<typeof vi.fn> }

        await wrapper.setProps({ timestamps: [0, 50, 100], values: [1, 2, 3] })

        expect(chart.setData).toHaveBeenCalled()
        // resetScales=false pins the viewport across the data swap (the invariant).
        expect(chart.setData.mock.calls.at(-1)![1]).toBe(false)
    })

    describe('with fake timers', () => {
        beforeEach(() => vi.useFakeTimers())
        afterEach(() => vi.useRealTimers())

        it('test_initial_render_does_not_emit_zoom', () => {
            const wrapper = mount(WaveformChart, { props: LONG_PROPS })
            vi.advanceTimersByTime(ZOOM_FETCH_DEBOUNCE_MS)

            expect(wrapper.emitted('zoom')).toBeUndefined()
        })

        it('test_single_user_zoom_emits_exactly_once_after_debounce', () => {
            const wrapper = mount(WaveformChart, { props: LONG_PROPS })
            // Rapid successive setScale calls (a single drag interaction) coalesce.
            chartOf().setScale('x', { min: 10, max: 40 })
            chartOf().setScale('x', { min: 10, max: 30 })

            expect(wrapper.emitted('zoom')).toBeUndefined()
            vi.advanceTimersByTime(ZOOM_FETCH_DEBOUNCE_MS)

            const emitted = wrapper.emitted('zoom')
            expect(emitted).toHaveLength(1)
            expect(emitted![0]).toEqual([10, 30])
        })

        it('test_emitted_zoom_subtracts_epoch_offset_and_reflects_floor', () => {
            const wrapper = mount(WaveformChart, {
                props: { ...LONG_PROPS, startEpoch: 1000 },
            })
            // Data lives at epoch [1000, 1100]; a 1 s drag floors to 5 s.
            chartOf().setScale('x', { min: 1010, max: 1011 })
            vi.advanceTimersByTime(ZOOM_FETCH_DEBOUNCE_MS)

            const emitted = wrapper.emitted('zoom')
            expect(emitted).toHaveLength(1)
            // Emitted values are session-relative (epoch subtracted) and floored to 5 s.
            const [start, end] = emitted![0] as [number, number]
            expect(start).toBeCloseTo(8, 10)
            expect(end).toBeCloseTo(13, 10)
        })

        it('test_setScaleX_same_window_skips_setScale_and_does_not_swallow_next_emit', () => {
            const wrapper = mount(WaveformChart, { props: LONG_PROPS })
            const vm = wrapper.vm as unknown as { setScaleX: (min: number, max: number) => void }
            const chart = mockState.lastChart as unknown as {
                scales: { x?: { min: number; max: number } }
                setScale: ((key: string, range: { min: number; max: number }) => void) & {
                    mock: { calls: unknown[] }
                }
            }

            // First programmatic zoom commits the window and arms+consumes isInitialRender.
            vm.setScaleX(10, 30)
            const callsAfterFirst = chart.setScale.mock.calls.length
            expect(chart.scales.x!.min).toBeCloseTo(10, 10)
            expect(chart.scales.x!.max).toBeCloseTo(30, 10)

            // Same window again: the idempotence guard returns before touching setScale.
            vm.setScaleX(10, 30)
            expect(chart.setScale.mock.calls.length).toBe(callsAfterFirst)

            // A genuine gesture still emits — the no-op call left no armed flag behind.
            chart.setScale('x', { min: 12, max: 28 })
            vi.advanceTimersByTime(ZOOM_FETCH_DEBOUNCE_MS)
            const emitted = wrapper.emitted('zoom')
            expect(emitted).toHaveLength(1)
            expect(emitted![0]).toEqual([12, 28])
        })
    })
})
