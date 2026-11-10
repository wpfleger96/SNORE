import { describe, expect, it, vi, beforeEach } from 'vitest'
import { mount, flushPromises } from '@vue/test-utils'

// uPlot is called at module scope (uPlot.sync(...)); mock it before importing the component.
vi.mock('uplot', () => {
    function UPlotMock(this: Record<string, unknown>) {
        this.destroy = vi.fn()
        this.setData = vi.fn()
        this.setSize = vi.fn()
        this.setScale = vi.fn()
        this.batch = vi.fn()
        this.over = document.createElement('div')
        this.scales = {}
        this.height = 240
        this.redraw = vi.fn()
    }
    ;(UPlotMock as unknown as Record<string, unknown>).sync = vi.fn(() => ({
        key: 'k',
        sub: vi.fn(),
        unsub: vi.fn(),
    }))
    ;(UPlotMock as unknown as Record<string, unknown>).pxRatio = 1
    return { default: UPlotMock }
})

vi.mock('@/api/waveforms')

// Stub child components that pull in uPlot or complex UI.
vi.mock('@/components/WaveformChart.vue', () => ({
    default: {
        name: 'WaveformChart',
        props: [
            'timestamps',
            'values',
            'unit',
            'label',
            'waveformType',
            'syncKey',
            'startEpoch',
            'refetching',
            'events',
            'bands',
        ],
        emits: ['zoom'],
        // Viewport methods the parent drives via chartRefs (no-ops for the stub).
        methods: {
            setScaleX() {},
            resetZoom() {},
        },
        template: '<div class="waveform-chart-stub" :data-refetching="refetching" />',
    },
}))

vi.mock('@/components/InfoHint.vue', () => ({
    default: { template: '<span class="info-hint-stub" />' },
}))

vi.mock('@/components/ui/select', () => ({
    Select: {
        template: '<div><slot /></div>',
        props: ['modelValue'],
        emits: ['update:modelValue'],
    },
    SelectTrigger: { template: '<div><slot /></div>' },
    SelectValue: { template: '<span />' },
    SelectContent: { template: '<div><slot /></div>' },
    SelectItem: { template: '<div><slot /></div>', props: ['value'] },
}))

vi.mock('@/components/ui/button', () => ({
    Button: { template: '<button v-bind="$attrs"><slot /></button>' },
}))

vi.mock('@lucide/vue', () => ({
    X: { template: '<svg class="icon-x" />' },
    Loader2: { template: '<svg class="loader2-stub animate-spin" />' },
    AlertTriangle: { template: '<svg class="icon-alert-triangle" />' },
}))

import { getWaveformData } from '@/api/waveforms'
import { Select } from '@/components/ui/select'
import MultiWaveformView from '@/components/MultiWaveformView.vue'
import { createWaveformCacheRegistry } from '@/utils/waveformCache'
import { BASE_MAX_POINTS } from '@/constants/waveform'
import type { WaveformDataResponse } from '@/types'

const mockGetWaveformData = vi.mocked(getWaveformData)

function makeResponse(overrides: Partial<WaveformDataResponse> = {}): WaveformDataResponse {
    return {
        timestamps: [0, 1, 2],
        values: [1.0, 1.5, 2.0],
        unit: 'L/min',
        downsampled: false,
        returned_samples: 3,
        sample_rate: 1,
        total_samples: 3,
        ...overrides,
    }
}

// A fresh registry per mount keeps cache state from leaking between tests.
function makeProps(overrides: Record<string, unknown> = {}) {
    return {
        sessionId: 42,
        availableTypes: ['flow', 'pressure'],
        initialTypes: ['flow'],
        startEpoch: 0,
        cacheRegistry: createWaveformCacheRegistry(),
        durationSec: 3600,
        ...overrides,
    }
}

beforeEach(() => {
    vi.resetAllMocks()
})

describe('MultiWaveformView', () => {
    it('test_initial_load_in_flight_shows_full_height_spinner_and_no_chart', async () => {
        // Arrange: deferred promise — load never resolves during this test
        let _resolve!: (v: WaveformDataResponse) => void
        const deferred = new Promise<WaveformDataResponse>((res) => {
            _resolve = res
        })
        mockGetWaveformData.mockReturnValue(deferred)

        // Act
        const wrapper = mount(MultiWaveformView, { props: makeProps() })
        await Promise.resolve() // allow onMounted to run

        // Assert
        expect(wrapper.find('.animate-spin').exists()).toBe(true)
        expect(wrapper.find('.waveform-chart-stub').exists()).toBe(false)

        // Cleanup
        _resolve(makeResponse())
        await flushPromises()
    })

    it('test_load_resolves_shows_chart_and_hides_spinner', async () => {
        // Arrange
        mockGetWaveformData.mockResolvedValue(makeResponse())

        // Act
        const wrapper = mount(MultiWaveformView, { props: makeProps() })
        await flushPromises()

        // Assert
        expect(wrapper.find('.waveform-chart-stub').exists()).toBe(true)
        expect(wrapper.find('.animate-spin').exists()).toBe(false)
    })

    it('test_zoomTo_keeps_chart_mounted_with_refetching_true_then_false_after_resolve', async () => {
        // Arrange: initial full-night load is inexact (returned == max_points), so a zoom into a
        // sparse sub-window misses the cache and triggers a refetch.
        mockGetWaveformData.mockResolvedValueOnce(makeResponse({ returned_samples: 2000 }))

        const wrapper = mount(MultiWaveformView, { props: makeProps() })
        await flushPromises()

        // Sanity: chart rendered
        expect(wrapper.find('.waveform-chart-stub').exists()).toBe(true)

        // Now deferred zoom refetch
        let resolveZoom!: (v: WaveformDataResponse) => void
        const zoomDeferred = new Promise<WaveformDataResponse>((res) => {
            resolveZoom = res
        })
        mockGetWaveformData.mockReturnValueOnce(zoomDeferred)

        // Act: call zoomTo via exposed
        const vm = wrapper.vm as unknown as { zoomTo: (s: number, e: number) => void }
        vm.zoomTo(10, 20)
        await Promise.resolve()

        // Assert: chart stays mounted, refetching=true, no full-height spinner
        expect(wrapper.find('.waveform-chart-stub').exists()).toBe(true)
        expect(wrapper.find('.waveform-chart-stub').attributes('data-refetching')).toBe('true')
        expect(wrapper.find('.animate-spin').exists()).toBe(false)

        // Resolve zoom and verify refetching clears
        resolveZoom(makeResponse())
        await flushPromises()

        expect(wrapper.find('.waveform-chart-stub').attributes('data-refetching')).toBe('false')
    })

    it('test_zoom_refetch_reject_keeps_chart_mounted_and_shows_banner', async () => {
        // Arrange: full-night loads resolve inexact (so the zoom misses the cache and the
        // background prefetch keeps the overview inexact); the windowed zoom fetch rejects.
        mockGetWaveformData.mockImplementation((_s, _t, params) =>
            params?.start_seconds !== undefined
                ? Promise.reject(new Error('timeout'))
                : Promise.resolve(makeResponse({ returned_samples: 10000 })),
        )

        const wrapper = mount(MultiWaveformView, { props: makeProps() })
        await flushPromises()

        // Act: trigger zoom via zoomTo
        const vm = wrapper.vm as unknown as { zoomTo: (s: number, e: number) => void }
        vm.zoomTo(10, 20)
        await flushPromises()

        // Assert: chart still rendered, error banner shown, no full-height error block
        expect(wrapper.find('.waveform-chart-stub').exists()).toBe(true)
        expect(wrapper.find('.h-60').exists()).toBe(false)
        expect(wrapper.text()).toContain('Failed to refresh waveform: timeout')
    })

    it('test_initial_load_rejects_shows_error_block_and_no_chart', async () => {
        // Arrange
        mockGetWaveformData.mockRejectedValue(new Error('server error'))

        // Act
        const wrapper = mount(MultiWaveformView, { props: makeProps() })
        await flushPromises()

        // Assert: error block shown, chart absent
        expect(wrapper.find('.waveform-chart-stub').exists()).toBe(false)
        expect(wrapper.text()).toContain('server error')
    })

    it('test_rezoom_into_cached_region_issues_no_fetches', async () => {
        // Arrange: exact full-night overview (returned < max_points) for both charts.
        mockGetWaveformData.mockResolvedValue(makeResponse())
        const wrapper = mount(MultiWaveformView, {
            props: makeProps({ initialTypes: ['flow', 'pressure'] }),
        })
        await flushPromises()
        expect(mockGetWaveformData).toHaveBeenCalledTimes(2) // one full-night fetch per chart
        mockGetWaveformData.mockClear()

        // Act: zoom into a sub-window fully covered by each exact overview.
        const vm = wrapper.vm as unknown as { zoomTo: (s: number, e: number) => void }
        vm.zoomTo(0.5, 1.5)
        await flushPromises()

        // Assert: served entirely from cache — no network across any chart.
        expect(mockGetWaveformData).not.toHaveBeenCalled()
    })

    it('test_first_zoom_into_new_region_fetches_once_per_chart', async () => {
        // Arrange: full-night loads resolve inexact (so a fresh region misses); windowed zoom
        // fetches resolve exact (so the zoom's own background prefetch resolves as hits and adds
        // no fetches). Post-zoom the only calls are the user-facing fetches, one per chart.
        mockGetWaveformData.mockImplementation((_s, _t, params) =>
            params?.start_seconds !== undefined
                ? Promise.resolve(makeResponse())
                : Promise.resolve(makeResponse({ returned_samples: 10000 })),
        )
        const wrapper = mount(MultiWaveformView, {
            props: makeProps({ initialTypes: ['flow', 'pressure'] }),
        })
        await flushPromises()
        mockGetWaveformData.mockClear() // discard initial loads and their background prefetch

        // Act: zoom into a fresh region uncovered by any exact chunk.
        const vm = wrapper.vm as unknown as { zoomTo: (s: number, e: number) => void }
        vm.zoomTo(10, 20)
        await flushPromises()

        // Assert: exactly one user-facing refetch per chart; the zoom's prefetch stayed in cache.
        expect(mockGetWaveformData).toHaveBeenCalledTimes(2)
    })

    it('test_type_switch_and_back_reuses_cached_full_night', async () => {
        // Arrange: exact overview cached for flow.
        mockGetWaveformData.mockResolvedValue(makeResponse())
        const wrapper = mount(MultiWaveformView, {
            props: makeProps({ initialTypes: ['flow'], availableTypes: ['flow', 'pressure'] }),
        })
        await flushPromises()
        expect(mockGetWaveformData).toHaveBeenCalledTimes(1) // flow full night

        const select = wrapper.findComponent(Select)

        // Act 1: switch to pressure — a new type, so it fetches once.
        select.vm.$emit('update:modelValue', 'pressure')
        await flushPromises()
        expect(mockGetWaveformData).toHaveBeenCalledTimes(2)
        mockGetWaveformData.mockClear()

        // Act 2: switch back to flow — its cache persisted, so no fetch.
        select.vm.$emit('update:modelValue', 'flow')
        await flushPromises()

        // Assert
        expect(mockGetWaveformData).not.toHaveBeenCalled()
    })

    it('test_prefetch_after_initial_load_serves_the_next_zoom_from_cache', async () => {
        // The initial full-night load is inexact (returned == BASE_MAX_POINTS), so the background
        // prefetch of the next zoom step fires. The prefetch fetches a denser full-night window
        // whose response is exact, upgrading the overview so the follow-up zoom is a cache hit.
        mockGetWaveformData.mockImplementation((_s, _t, params) =>
            Promise.resolve(
                params?.max_points === BASE_MAX_POINTS
                    ? makeResponse({ returned_samples: BASE_MAX_POINTS })
                    : makeResponse(),
            ),
        )
        const wrapper = mount(MultiWaveformView, { props: makeProps({ initialTypes: ['flow'] }) })
        await flushPromises()

        // A background prefetch fetch (denser than the initial load) must have been issued.
        const prefetched = mockGetWaveformData.mock.calls.some(
            (c) => (c[2]?.max_points ?? 0) > BASE_MAX_POINTS,
        )
        expect(prefetched).toBe(true)
        mockGetWaveformData.mockClear()

        // Act: take the zoom-in step the prefetch warmed (centered, half the full-night width).
        const vm = wrapper.vm as unknown as { zoomTo: (s: number, e: number) => void }
        vm.zoomTo(900, 2700)
        await flushPromises()

        // Assert: no user-facing fetch — the step resolves entirely from the prefetched cache.
        expect(mockGetWaveformData).not.toHaveBeenCalled()
    })

    it('test_unmount_aborts_in_flight_fetch_and_stores_nothing', async () => {
        // Model axios: an aborted request rejects (CanceledError), so its response is never stored.
        let capturedSignal: AbortSignal | undefined
        mockGetWaveformData.mockImplementation(
            (_s, _t, _p, signal) =>
                new Promise<WaveformDataResponse>((_resolve, reject) => {
                    capturedSignal = signal as AbortSignal
                    signal?.addEventListener('abort', () =>
                        reject(Object.assign(new Error('canceled'), { name: 'CanceledError' })),
                    )
                }),
        )

        const registry = createWaveformCacheRegistry()
        const wrapper = mount(MultiWaveformView, {
            props: makeProps({ cacheRegistry: registry, initialTypes: ['flow'] }),
        })
        await Promise.resolve() // let onMounted → loadChart issue the in-flight fetch

        expect(capturedSignal).toBeDefined()
        expect(capturedSignal!.aborted).toBe(false)

        // Unmount mid-flight: onBeforeUnmount aborts the fetch, which rejects and never stores —
        // so a late response cannot land the previous session's chunk into the cleared cache.
        wrapper.unmount()
        await flushPromises()

        expect(capturedSignal!.aborted).toBe(true)
        expect(registry.getCache('flow').resolve(0, 3600, 3600).slice).toBeNull()
    })

    it('test_removing_a_chart_aborts_its_pending_prefetch', async () => {
        // Capture the AbortSignal handed to each background prefetch fetch. The expanded prefetch
        // window requests more than BASE_MAX_POINTS; the initial load requests exactly BASE.
        const prefetchSignals = new Map<string, AbortSignal>()
        mockGetWaveformData.mockImplementation((_s, type, params, signal) => {
            if (signal && (params?.max_points ?? 0) > BASE_MAX_POINTS) {
                prefetchSignals.set(type, signal)
            }
            // Inexact full-night overview so the initial-load prefetch fires.
            return Promise.resolve(makeResponse({ returned_samples: BASE_MAX_POINTS }))
        })

        const wrapper = mount(MultiWaveformView, {
            props: makeProps({ initialTypes: ['flow', 'pressure'] }),
        })
        await flushPromises()

        const flowPrefetch = prefetchSignals.get('flow')
        const pressurePrefetch = prefetchSignals.get('pressure')
        expect(flowPrefetch).toBeDefined()
        expect(pressurePrefetch).toBeDefined()
        expect(flowPrefetch!.aborted).toBe(false)

        // Act: remove the flow chart (its remove button is the first one).
        await wrapper.findAll('.chart-remove')[0]!.trigger('click')

        // Assert: flow's pending prefetch was aborted; the surviving chart's prefetch is untouched.
        expect(flowPrefetch!.aborted).toBe(true)
        expect(pressurePrefetch!.aborted).toBe(false)
    })

    it('test_large_leak_bands_reach_only_the_leak_chart', async () => {
        mockGetWaveformData.mockResolvedValue(makeResponse())
        const bands = [{ start: 40, end: 60 }]

        const wrapper = mount(MultiWaveformView, {
            props: makeProps({
                availableTypes: ['flow', 'leak'],
                initialTypes: ['flow', 'leak'],
                largeLeakBands: bands,
            }),
        })
        await flushPromises()

        const charts = wrapper.findAllComponents({ name: 'WaveformChart' })
        const bandsByType = Object.fromEntries(
            charts.map((c) => [c.props('waveformType'), c.props('bands')]),
        )
        expect(bandsByType).toEqual({ flow: undefined, leak: bands })
    })
})
