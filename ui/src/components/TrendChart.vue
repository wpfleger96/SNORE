<template>
    <div ref="containerRef" class="trend-chart" />
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, watch } from 'vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { useDarkMode } from '@/composables/useDarkMode'
import { parseLocalDate } from '@/utils/formatting'

const { isDark } = useDarkMode()

interface Dataset {
    label: string
    values: (number | null)[]
    color: string
}

// A drawn series: `shown` is what its legend reports, which is the component's own value when
// `values` holds a stack's running total.
interface DrawnSeries extends Dataset {
    shown: (number | null)[]
}

const props = defineProps<{
    labels: string[]
    datasets: Dataset[]
    height?: number
    syncKey?: uPlot.SyncPubSub
    // Stacks the datasets as bars, the first at the bottom.
    stacked?: boolean
    // Formats legend values and y-axis ticks, which land on whole hours (for clock times).
    formatValue?: (value: number) => string
}>()

const containerRef = ref<HTMLDivElement>()
let chart: uPlot | null = null

function axisColors() {
    return isDark.value ? { axis: '#a1a1aa', grid: '#27272a' } : { axis: '#888', grid: '#eee' }
}

function legendText(value: number | null): string {
    if (value == null) return '--'
    return props.formatValue ? props.formatValue(value) : String(Number(value.toFixed(2)))
}

// Each bar is drawn from zero, so a stack is its running totals drawn tallest first, each shorter
// total painted over the one before. A missing component adds nothing to its period's stack, and
// a period with no component has no bar.
function stackedSeries(): DrawnSeries[] {
    const running = props.labels.map(() => 0)
    const hasAny = props.labels.map((_, i) => props.datasets.some((ds) => ds.values[i] != null))
    return props.datasets
        .map((ds) => ({
            ...ds,
            shown: ds.values,
            values: ds.values.map((v, i) => (hasAny[i] ? (running[i] += v ?? 0) : null)),
        }))
        .reverse()
}

// Half the smallest spacing between periods (12 hours for a single period), so padding the x
// scale by it on each side draws the first and last bars whole.
function halfPeriod(timestamps: number[]): number {
    const gaps = timestamps.slice(1).map((t, i) => t - timestamps[i])
    return gaps.length ? Math.min(...gaps) / 2 : 43200
}

// Pads the data's extent by half a period; any other range (a zoom, a redraw of the padded range)
// is kept as given, so the padding never compounds.
function paddedXRange(timestamps: number[]): uPlot.Range.Function {
    const first = timestamps[0]
    const last = timestamps[timestamps.length - 1]
    const pad = halfPeriod(timestamps)
    return (_u, min, max) =>
        min === first && (max === last || first === last) ? [first - pad, last + pad] : [min, max]
}

function createChart(): void {
    if (!containerRef.value || !props.labels.length) return

    chart?.destroy()

    const colors = axisColors()
    const timestamps = props.labels.map((d) => parseLocalDate(d).getTime() / 1000)
    const drawn: DrawnSeries[] = props.stacked
        ? stackedSeries()
        : props.datasets.map((ds) => ({ ...ds, shown: ds.values }))
    const format = props.formatValue
    const width = containerRef.value.clientWidth || 800

    const opts: uPlot.Options = {
        width,
        height: props.height ?? 280,
        cursor: {
            sync: props.syncKey ? { key: props.syncKey.key } : undefined,
            drag: { x: true, y: false, setScale: true },
        },
        scales: props.stacked
            ? {
                  x: { time: true, range: paddedXRange(timestamps) },
                  y: { range: (_u, _min, max) => [0, max > 0 ? max * 1.1 : 1] },
              }
            : { x: { time: true } },
        axes: [
            {
                space: 80,
                stroke: colors.axis,
                grid: { stroke: colors.grid },
                ticks: { stroke: colors.axis },
            },
            {
                size: 60,
                stroke: colors.axis,
                grid: { stroke: colors.grid },
                ticks: { stroke: colors.axis },
                ...(format
                    ? {
                          incrs: [1, 2, 3, 6, 12],
                          values: (_u: uPlot, splits: number[]) => splits.map(format),
                      }
                    : {}),
            },
        ],
        series: [
            {},
            ...drawn.map((ds) => ({
                label: ds.label,
                stroke: ds.color,
                ...(props.stacked
                    ? { fill: ds.color, paths: uPlot.paths.bars!({ size: [0.6, 64] }) }
                    : { width: 2, spanGaps: true, points: { size: 4 } }),
                ...(props.stacked || format
                    ? {
                          value: (_u: uPlot, _v: unknown, _si: number, idx: number | null) =>
                              legendText(idx == null ? null : ds.shown[idx]),
                      }
                    : {}),
            })),
        ],
    }

    const data: uPlot.AlignedData = [
        timestamps,
        ...drawn.map((ds) => ds.values.map((v) => v ?? undefined) as (number | undefined)[]),
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
})

watch(
    () =>
        [
            props.labels,
            props.datasets,
            props.height,
            props.syncKey,
            props.stacked,
            props.formatValue,
        ] as const,
    () => createChart(),
    { deep: true },
)

watch(isDark, () => createChart())
</script>

<style scoped>
.trend-chart {
    width: 100%;
}
</style>
