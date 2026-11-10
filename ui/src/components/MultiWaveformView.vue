<template>
    <div class="flex flex-col gap-4">
        <div
            v-for="(chart, idx) in charts"
            :key="chart.id"
            class="border border-border rounded-lg p-3 bg-card"
        >
            <div class="flex items-center justify-between mb-2 max-md:flex-wrap">
                <div class="flex items-center gap-2">
                    <Select
                        :model-value="chart.type"
                        @update:model-value="(v) => updateChartType(idx, v as string)"
                    >
                        <SelectTrigger class="w-[180px] h-8 text-sm">
                            <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem
                                v-for="opt in typeOptions"
                                :key="opt.value"
                                :value="opt.value"
                            >
                                {{ opt.label }}
                            </SelectItem>
                        </SelectContent>
                    </Select>
                    <InfoHint :glossary-key="WAVEFORM_GLOSSARY_MAP[chart.type as WaveformType]" />
                </div>
                <Button
                    v-if="charts.length > 1"
                    variant="ghost"
                    size="icon"
                    class="chart-remove text-destructive hover:text-destructive"
                    @click="removeChart(idx)"
                >
                    <X class="h-4 w-4" />
                </Button>
            </div>
            <!-- Full-height spinner only before first data; refetches keep the chart mounted with a corner spinner (:refetching) -->
            <div
                v-if="chart.loading && !chart.data"
                class="h-60 flex items-center justify-center gap-2 text-muted-foreground"
            >
                <Loader2 class="h-4 w-4 animate-spin" />
                Loading {{ WAVEFORM_LABELS[chart.type] ?? chart.type }}...
            </div>
            <div
                v-else-if="chart.error && !chart.data"
                class="h-60 flex items-center justify-center gap-2 text-destructive"
            >
                <AlertTriangle class="h-4 w-4" />
                {{ chart.error }}
            </div>
            <WaveformChart
                v-else-if="chart.data"
                :ref="(el) => setChartRef(idx, el)"
                :timestamps="chart.data.timestamps"
                :values="chart.data.values"
                :unit="chart.data.unit"
                :label="WAVEFORM_LABELS[chart.type] ?? chart.type"
                :waveform-type="chart.type"
                :events="chart.type === 'flow' ? events : undefined"
                :bands="chart.type === 'leak' ? largeLeakBands : undefined"
                :sync-key="syncKey"
                :start-epoch="startEpoch"
                :refetching="chart.loading"
                @zoom="onZoom"
            />
            <div
                v-if="chart.data && chart.error && !chart.loading"
                class="mt-1 text-sm text-destructive"
            >
                Failed to refresh waveform: {{ chart.error }}
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted, onBeforeUnmount, nextTick } from 'vue'
import uPlot from 'uplot'
import { X, Loader2, AlertTriangle } from '@lucide/vue'
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import WaveformChart from './WaveformChart.vue'
import InfoHint from '@/components/InfoHint.vue'
import { getWaveformData } from '@/api/waveforms'
import { prefetchAdjacentWindows } from '@/utils/waveformPrefetch'
import { buildWaveformParams } from '@/utils/waveformCache'
import { WAVEFORM_LABELS, WAVEFORM_GLOSSARY_MAP } from '@/types'
import type { ChartBand, EventItem, WaveformType } from '@/types'
import type { WaveformCacheRegistry, WaveformSlice } from '@/utils/waveformCache'

const props = defineProps<{
    sessionId: number
    availableTypes: string[]
    events?: EventItem[]
    // Large Leak spans in session-relative seconds, banded on the leak chart.
    largeLeakBands?: ChartBand[]
    initialTypes?: string[]
    startEpoch: number
    cacheRegistry: WaveformCacheRegistry
    durationSec: number
}>()

const emit = defineEmits<{
    zoom: [startSec: number, endSec: number]
}>()

interface ChartState {
    id: number
    type: string
    data: WaveformSlice | null
    loading: boolean
    error: string | null
}

let nextId = 0
const syncKey = uPlot.sync(`waveform-sync-${crypto.randomUUID()}`)
const charts = ref<ChartState[]>([])
const chartRefs = ref<(InstanceType<typeof WaveformChart> | null)[]>([])
const chartAborts = new Map<number, AbortController>()
const chartPrefetch = new Map<number, AbortController>()

const typeOptions = computed(() =>
    props.availableTypes.map((t) => ({ value: t, label: WAVEFORM_LABELS[t] ?? t })),
)

function setChartRef(idx: number, el: unknown): void {
    chartRefs.value[idx] = el as InstanceType<typeof WaveformChart> | null
}

// Warm the adjacent zoom windows for this chart in the background so the next step is a cache hit.
// Fire-and-forget: never touches chart.data/loading/error. Supersedes any prior prefetch.
function kickChartPrefetch(chart: ChartState, startSec: number, endSec: number): void {
    chartPrefetch.get(chart.id)?.abort()
    const controller = new AbortController()
    chartPrefetch.set(chart.id, controller)
    prefetchAdjacentWindows({
        cache: props.cacheRegistry.getCache(chart.type),
        sessionId: props.sessionId,
        waveformType: chart.type,
        startSec,
        endSec,
        durationSec: props.durationSec,
        signal: controller.signal,
    })
}

async function loadChart(chart: ChartState, startSec?: number, endSec?: number): Promise<void> {
    // A newer request (even a cache hit) supersedes any in-flight fetch and pending prefetch.
    chartAborts.get(chart.id)?.abort()
    chartPrefetch.get(chart.id)?.abort()
    chartPrefetch.delete(chart.id)
    const start = startSec ?? 0
    const end = endSec ?? props.durationSec
    const cache = props.cacheRegistry.getCache(chart.type)

    const resolved = cache.resolve(start, end, props.durationSec)
    if (resolved.slice) chart.data = resolved.slice
    chart.error = null
    if (resolved.hit || !resolved.fetchWindow) {
        chart.loading = false
        chartAborts.delete(chart.id)
        kickChartPrefetch(chart, start, end)
        return
    }

    const thisController = new AbortController()
    chartAborts.set(chart.id, thisController)
    const fetchWindow = resolved.fetchWindow

    chart.loading = true
    try {
        const response = await getWaveformData(
            props.sessionId,
            chart.type,
            buildWaveformParams(fetchWindow, props.durationSec),
            thisController.signal,
        )
        // Superseded (or unmounted) → never store: a stale response must not land a chunk into a
        // cache the parent may have cleared on session change (matches useWaveformWindow.runFetch).
        if (chartAborts.get(chart.id) === thisController) {
            cache.store(fetchWindow, response, props.durationSec)
            chart.data = cache.resolve(start, end, props.durationSec).slice
            kickChartPrefetch(chart, start, end)
        }
    } catch (err: unknown) {
        if (err instanceof Error && err.name !== 'CanceledError') {
            chart.error = err.message
        }
    } finally {
        if (chartAborts.get(chart.id) === thisController) {
            chart.loading = false
        }
    }
}

function addChart(type: string): void {
    if (charts.value.length >= 4) return
    charts.value.push({ id: nextId++, type, data: null, loading: false, error: null })
    void loadChart(charts.value[charts.value.length - 1]!)
}

function removeChart(idx: number): void {
    const chart = charts.value[idx]
    if (chart) {
        chartAborts.get(chart.id)?.abort()
        chartAborts.delete(chart.id)
        chartPrefetch.get(chart.id)?.abort()
        chartPrefetch.delete(chart.id)
    }
    charts.value.splice(idx, 1)
    chartRefs.value.splice(idx, 1)
}

function updateChartType(idx: number, newType: string): void {
    const chart = charts.value[idx]
    if (!chart) return
    // loadChart's top already aborts+drops the outgoing type's in-flight fetch and prefetch.
    chart.type = newType
    chart.data = null
    chart.error = null
    void loadChart(chart)
}

function onZoom(startSec: number, endSec: number): void {
    // Data swaps no longer rescale; propagate the viewport to every chart explicitly. The origin
    // chart already sits at this window, so its setScaleX is a no-op (idempotence guard).
    chartRefs.value.forEach((r) => r?.setScaleX(startSec, endSec))
    for (const chart of charts.value) {
        void loadChart(chart, startSec, endSec)
    }
    emit('zoom', startSec, endSec)
}

onMounted(() => {
    const types = props.initialTypes ?? [props.availableTypes[0] ?? 'flow']
    for (const t of types) addChart(t)
})

onBeforeUnmount(() => {
    chartAborts.forEach((c) => c.abort())
    chartPrefetch.forEach((c) => c.abort())
})

defineExpose({
    addChart,
    async resetZoom() {
        for (const chart of charts.value) {
            void loadChart(chart)
        }
        // resetZoom rescales to the data extent — valid only after the overview slice flushes to
        // props; loadChart resolves the overview synchronously but the prop update is async.
        await nextTick()
        chartRefs.value.forEach((r) => r?.resetZoom())
    },
    zoomTo(startSec: number, endSec: number) {
        // Move every chart's viewport explicitly (data swaps no longer rescale).
        chartRefs.value.forEach((r) => r?.setScaleX(startSec, endSec))
        for (const chart of charts.value) {
            void loadChart(chart, startSec, endSec)
        }
    },
    get chartCount() {
        return charts.value.length
    },
    chartTypes(): string[] {
        return charts.value.map((c) => c.type)
    },
})
</script>

<style scoped>
@media (max-width: 767.98px) {
    /* shadcn size="icon" is layered (36px) and loses to scoped CSS, so the touch floor lives here */
    .chart-remove {
        min-height: var(--tap-target);
        min-width: var(--tap-target);
    }
}
</style>
