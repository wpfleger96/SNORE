<template>
    <section v-if="segments?.length" class="section-card">
        <h2>Running AHI <InfoHint glossary-key="running_ahi" /></h2>
        <UPlotChart :data="data" :options="options" />
    </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type uPlot from 'uplot'
import InfoHint from '@/components/InfoHint.vue'
import UPlotChart, { type AxisColors } from '@/components/UPlotChart.vue'
import { formatWallClockTime, parseLocalDate } from '@/utils/formatting'
import type { RunningAhiSegment } from '@/types'

const props = defineProps<{ segments?: RunningAhiSegment[] | null }>()

const STEP_SECONDS = 60

// One point per minute; a null midway between segments breaks the line over
// mask-off time.  A segment that starts before the previous one ends (sessions
// can overlap) cuts it short, since uPlot needs ascending x values.
const data = computed<uPlot.AlignedData>(() => {
    const xs: number[] = []
    const ys: (number | null)[] = []
    for (const segment of props.segments ?? []) {
        const start = parseLocalDate(segment.start).getTime() / 1000
        while ((xs.at(-1) ?? -Infinity) >= start) {
            xs.pop()
            ys.pop()
        }
        const previous = xs.at(-1)
        if (previous !== undefined) {
            xs.push((previous + start) / 2)
            ys.push(null)
        }
        segment.values.forEach((value, i) => {
            xs.push(start + i * STEP_SECONDS)
            ys.push(value)
        })
    }
    return [xs, ys]
})

function options(colors: AxisColors): Omit<uPlot.Options, 'width'> {
    const axis = {
        stroke: colors.axis,
        grid: { stroke: colors.grid },
        ticks: { stroke: colors.axis },
    }
    return {
        height: 220,
        cursor: { drag: { x: true, y: false, setScale: true } },
        scales: { x: { time: true }, y: { range: (_u, _min, max) => [0, Math.max(5, max)] } },
        axes: [
            {
                ...axis,
                space: 80,
                values: (_u, vals, _axis, _space, foundIncr) =>
                    vals.map((v) => (v == null ? '' : formatWallClockTime(v, foundIncr))),
            },
            { ...axis, size: 50, label: 'Events/h' },
        ],
        series: [
            { value: (_u, v) => (v == null ? '--' : formatWallClockTime(v, 0)) },
            { label: 'Running AHI', stroke: 'rgb(220, 38, 38)', width: 2, points: { show: false } },
        ],
    }
}
</script>
