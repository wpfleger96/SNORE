<template>
    <section v-if="tap?.pressures.length" class="section-card">
        <h2>Time at Pressure <InfoHint glossary-key="time_at_pressure" /></h2>
        <UPlotChart :data="data" :options="options" />
    </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import type uPlot from 'uplot'
import InfoHint from '@/components/InfoHint.vue'
import UPlotChart, { type AxisColors } from '@/components/UPlotChart.vue'
import { EVENT_COLORS, type TimeAtPressure } from '@/types'

const props = defineProps<{ tap?: TimeAtPressure | null }>()

const toMinutes = (seconds: number[]) => seconds.map((s) => s / 60)

const eventTypes = computed(() => Object.keys(props.tap?.event_counts ?? {}))

// Minutes per bucket for pressure and EPAP, then one point series per event
// type on its own scale; a bucket with no events of a type plots nothing.
const data = computed<uPlot.AlignedData>(() => {
    const tap = props.tap
    if (!tap) return [[]]
    return [
        tap.pressures,
        toMinutes(tap.pressure_seconds),
        ...(tap.epap_seconds ? [toMinutes(tap.epap_seconds)] : []),
        ...eventTypes.value.map((type) =>
            (tap.event_counts[type] ?? []).map((count) => (count > 0 ? count : null)),
        ),
    ]
})

function options(colors: AxisColors): Omit<uPlot.Options, 'width'> {
    const axis = {
        stroke: colors.axis,
        grid: { stroke: colors.grid },
        ticks: { stroke: colors.axis },
    }
    const minutes = { scale: 'min', width: 2, points: { show: false } }
    return {
        height: 240,
        cursor: { drag: { x: true, y: false, setScale: true } },
        scales: { x: { time: false } },
        axes: [
            { ...axis, label: 'cmH₂O', values: (_u, vals) => vals.map((v) => v.toFixed(1)) },
            { ...axis, scale: 'min', size: 50, label: 'Minutes' },
            { ...axis, scale: 'events', side: 1, size: 40, label: 'Events', grid: { show: false } },
        ],
        series: [
            { value: (_u, v) => (v == null ? '--' : `${v.toFixed(1)} cmH₂O`) },
            { ...minutes, label: 'Pressure', stroke: 'rgb(37, 99, 235)' },
            ...(props.tap?.epap_seconds
                ? [{ ...minutes, label: 'EPAP', stroke: 'rgb(13, 148, 136)' }]
                : []),
            ...eventTypes.value.map((type) => ({
                label: type,
                scale: 'events',
                stroke: (EVENT_COLORS[type] ?? 'rgba(100, 100, 100, 0.25)').replace(
                    /[\d.]+\)$/,
                    '1)',
                ),
                paths: () => null,
                points: { show: true, size: 7 },
            })),
        ],
    }
}
</script>
