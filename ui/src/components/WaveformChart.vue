<template>
    <div class="relative">
        <div ref="containerRef" class="waveform-chart" />
        <div v-if="refetching" class="absolute top-2 right-2 text-muted-foreground">
            <Loader2 class="h-4 w-4 animate-spin" />
        </div>
        <p v-if="events?.length" class="mt-1 text-xs text-muted-foreground">
            Shaded bands: device-scored events
        </p>
    </div>
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, watch, nextTick } from 'vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { Loader2 } from '@lucide/vue'
import { EVENT_COLORS, EVENT_SOLID_COLORS } from '@/types'
import type { EventItem } from '@/types'
import { useDarkMode } from '@/composables/useDarkMode'
import { formatWallClockTime } from '@/utils/formatting'
import { MIN_ZOOM_WINDOW_SEC, ZOOM_FETCH_DEBOUNCE_MS } from '@/constants/waveform'

const { isDark } = useDarkMode()

const props = defineProps<{
    timestamps: number[]
    values: number[]
    unit: string
    label: string
    startEpoch: number
    waveformType?: string
    events?: EventItem[]
    syncKey?: uPlot.SyncPubSub
    refetching?: boolean
}>()

const emit = defineEmits<{
    zoom: [startSec: number, endSec: number]
}>()

// X-scale epoch convention: the x scale runs in wall-clock epoch seconds (`time: true`).
// Every inbound x position must add `props.startEpoch` to a session-relative offset,
// and every outbound value (emits/exposed methods) must subtract it to restore
// session-relative seconds. Future canvas-drawing features must follow the same rule.
//
// Viewport invariant: data swaps NEVER move the viewport. The data watch calls
// setData(..., false) so a cache miss's interim slice cannot rescale the x range; every
// viewport change comes from a user gesture or an explicit setScaleX/resetZoom call.

const containerRef = ref<HTMLDivElement>()
let chart: uPlot | null = null
let debounceTimer: ReturnType<typeof setTimeout> | null = null
let isInitialRender = true

// Enforce the 5 s max-zoom floor for every zoom path (drag, wheel, programmatic
// setScaleX). uPlot invokes the x scale's range fn on each setScale before
// committing, so flooring here is the single choke point: the setScale hook then
// reads already-clamped min/max and emits once, with no re-entrant setScale.
// All values are wall-clock epoch seconds, matching the x scale.
function clampSpan(min: number, max: number): [number, number] {
    const ts = props.timestamps
    if (!ts.length || min == null || max == null) return [min, max]
    const dataMin = ts[0] + props.startEpoch
    const dataMax = ts[ts.length - 1] + props.startEpoch
    // A session shorter than the floor can only ever show its full span.
    const floor = Math.min(MIN_ZOOM_WINDOW_SEC, dataMax - dataMin)
    if (max - min >= floor) return [min, max]
    // Expand around the window center to `floor` wide, then shift the window so it
    // stays within [dataMin, dataMax]. Since floor <= dataSpan the window fits, so
    // the two shifts never conflict.
    const center = (min + max) / 2
    let lo = center - floor / 2
    let hi = center + floor / 2
    if (lo < dataMin) {
        hi += dataMin - lo
        lo = dataMin
    }
    if (hi > dataMax) {
        lo -= hi - dataMax
        hi = dataMax
    }
    return [lo, hi]
}

function buildEventPlugin(): uPlot.Plugin {
    return {
        hooks: {
            drawClear: [
                (u: uPlot) => {
                    if (!props.events?.length) return
                    const ctx = u.ctx
                    const { left, top, width, height } = u.bbox

                    ctx.save()
                    ctx.beginPath()
                    ctx.rect(left, top, width, height)
                    ctx.clip()
                    for (const evt of props.events) {
                        const color = EVENT_COLORS[evt.event_type]
                        if (!color) continue

                        const x0 = u.valToPos(evt.offset_seconds + props.startEpoch, 'x', true)
                        const x1 = u.valToPos(
                            evt.offset_seconds + evt.duration_seconds + props.startEpoch,
                            'x',
                            true,
                        )

                        if (x1 < left || x0 > left + width) continue

                        const cx0 = Math.max(x0, left)
                        const cx1 = Math.min(x1, left + width)
                        const bandW = Math.max(cx1 - cx0, 2 * uPlot.pxRatio)

                        ctx.fillStyle = color
                        ctx.fillRect(cx0, top, bandW, height)
                    }
                    ctx.restore()
                },
            ],
            draw: [
                (u: uPlot) => {
                    if (!props.events?.length) return
                    const ctx = u.ctx
                    const { left, top, width, height } = u.bbox
                    const margin = 4 * uPlot.pxRatio
                    const fontSize = Math.round(11 * uPlot.pxRatio)

                    ctx.save()
                    ctx.beginPath()
                    ctx.rect(left, top, width, height)
                    ctx.clip()
                    ctx.font = `bold ${fontSize}px sans-serif`
                    ctx.textAlign = 'center'
                    ctx.textBaseline = 'middle'

                    // Alternate labels top/bottom to avoid overlap when events cluster.
                    // Index parity (not drawn-count parity) keeps each event's label on
                    // the same side across pan/zoom, since events arrive sorted by start_time.
                    for (let i = 0; i < props.events.length; i++) {
                        const evt = props.events[i]
                        const solidColor = EVENT_SOLID_COLORS[evt.event_type]
                        if (!solidColor) continue

                        const x0 = u.valToPos(evt.offset_seconds + props.startEpoch, 'x', true)
                        const x1 = u.valToPos(
                            evt.offset_seconds + evt.duration_seconds + props.startEpoch,
                            'x',
                            true,
                        )

                        if (x1 < left || x0 > left + width) continue

                        const cx = (Math.max(x0, left) + Math.min(x1, left + width)) / 2

                        ctx.strokeStyle = solidColor
                        ctx.lineWidth = uPlot.pxRatio
                        ctx.beginPath()
                        ctx.moveTo(cx, top)
                        ctx.lineTo(cx, top + height)
                        ctx.stroke()

                        const labelY =
                            i % 2 === 0
                                ? top + margin + fontSize / 2
                                : top + height - margin - fontSize / 2

                        ctx.fillStyle = solidColor
                        // Rotate -90° about the label anchor; absolute matrix, so no save/restore needed.
                        ctx.setTransform(0, -1, 1, 0, cx, labelY)
                        ctx.fillText(evt.event_type, 0, 0)
                        ctx.setTransform(1, 0, 0, 1, 0, 0)
                    }

                    ctx.restore()
                },
            ],
        },
    }
}

function chartColors() {
    return isDark.value
        ? { axis: '#a1a1aa', grid: '#27272a', series: '#60a5fa', fill: 'rgba(96, 165, 250, 0.06)' }
        : { axis: '#888', grid: '#eee', series: '#2563eb', fill: 'rgba(37, 99, 235, 0.05)' }
}

function createChart(): void {
    if (!containerRef.value || !props.timestamps.length) return

    chart?.destroy()
    isInitialRender = true

    const colors = chartColors()
    const width = containerRef.value.clientWidth || 800
    const height = 240

    const opts: uPlot.Options = {
        width,
        height,
        plugins: [buildEventPlugin()],
        cursor: {
            sync: props.syncKey ? { key: props.syncKey.key } : undefined,
            drag: { x: true, y: false, setScale: true },
        },
        scales: {
            x: { time: true, range: (_u: uPlot, min: number, max: number) => clampSpan(min, max) },
            // Pin y-axis for channels that have a known fixed range so a quiet
            // night doesn't auto-scale to a misleadingly wide or narrow extent.
            y:
                props.waveformType === 'fl'
                    ? { min: 0, max: 1 }
                    : props.waveformType === 'snore'
                      ? { min: 0, max: 5 }
                      : {},
        },
        axes: [
            {
                values: (
                    _u: uPlot,
                    vals: number[],
                    _axisIdx: number,
                    _space: number,
                    foundIncr: number,
                ) => vals.map((v) => (v == null ? '' : formatWallClockTime(v, foundIncr))),
                space: 90,
                stroke: colors.axis,
                grid: { stroke: colors.grid },
                ticks: { stroke: colors.axis },
            },
            {
                label: `${props.label} (${props.unit})`,
                size: 70,
                stroke: colors.axis,
                grid: { stroke: colors.grid },
                ticks: { stroke: colors.axis },
            },
        ],
        series: [
            {
                value: (_u: uPlot, v: number | null) =>
                    v == null ? '--' : formatWallClockTime(v, 0),
            },
            {
                label: props.label,
                stroke: colors.series,
                width: 1,
                fill: colors.fill,
                // uPlot auto-shows point markers when visible samples are sparse, which
                // makes the interim cache slice flash dots until the denser fetch lands.
                points: { show: false },
            },
        ],
        hooks: {
            setScale: [
                (u: uPlot, scaleKey: string) => {
                    if (scaleKey !== 'x') return
                    // Skip the initial render scale-set
                    if (isInitialRender) {
                        isInitialRender = false
                        return
                    }
                    const min = u.scales.x.min
                    const max = u.scales.x.max
                    if (min == null || max == null) return

                    if (debounceTimer) clearTimeout(debounceTimer)
                    debounceTimer = setTimeout(() => {
                        emit('zoom', min - props.startEpoch, max - props.startEpoch)
                    }, ZOOM_FETCH_DEBOUNCE_MS)
                },
            ],
        },
    }

    const data: uPlot.AlignedData = [
        props.timestamps.map((t) => t + props.startEpoch),
        props.values,
    ]
    chart = new uPlot(opts, data, containerRef.value)
}

function handleResize(): void {
    if (!chart || !containerRef.value) return
    chart.setSize({ width: containerRef.value.clientWidth, height: chart.height })
}

let resizeObserver: ResizeObserver | null = null

onMounted(() => {
    createChart()
    resizeObserver = new ResizeObserver(handleResize)
    if (containerRef.value) resizeObserver.observe(containerRef.value)
})

onBeforeUnmount(() => {
    chart?.destroy()
    chart = null
    resizeObserver?.disconnect()
    if (debounceTimer) clearTimeout(debounceTimer)
})

// Update data in-place when timestamps/values change (avoids canvas flicker).
// resetScales=false pins the viewport (see invariant above): the interim cache slice never
// rescales the x range, so no setScale fires — hence isInitialRender must NOT be armed here,
// or it would swallow the next genuine gesture's zoom emit.
watch(
    () => [props.timestamps, props.values] as const,
    async ([ts, vals]) => {
        if (!ts.length) return
        if (!chart) {
            await nextTick()
            createChart()
            return
        }
        chart.setData([ts.map((t) => t + props.startEpoch), vals], false)
    },
)

// Redraw when events change (overlay update)
watch(
    () => props.events,
    () => chart?.redraw(),
    { deep: false },
)

watch(isDark, () => createChart())
watch(
    () => props.startEpoch,
    () => createChart(),
)

defineExpose({
    resetZoom() {
        if (!chart || !props.timestamps.length) return
        isInitialRender = true
        chart.setScale('x', {
            min: props.timestamps[0] + props.startEpoch,
            max: props.timestamps[props.timestamps.length - 1] + props.startEpoch,
        })
    },
    setScaleX(min: number, max: number) {
        if (!chart) return
        const targetMin = min + props.startEpoch
        const targetMax = max + props.startEpoch
        // Idempotence: if the x scale already sits at the target (e.g. the origin chart of a drag
        // receiving its own window back), skip setScale so isInitialRender is not left double-armed
        // to swallow a later genuine gesture.
        const cur = chart.scales.x
        if (
            cur.min != null &&
            cur.max != null &&
            Math.abs(cur.min - targetMin) < 1e-6 &&
            Math.abs(cur.max - targetMax) < 1e-6
        ) {
            return
        }
        isInitialRender = true
        chart.setScale('x', { min: targetMin, max: targetMax })
    },
})
</script>

<style scoped>
.waveform-chart {
    width: 100%;
}

.waveform-chart :deep(.uplot) {
    width: 100% !important;
}
</style>
