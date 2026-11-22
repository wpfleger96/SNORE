<template>
    <div ref="containerRef" class="uplot-chart" />
</template>

<script setup lang="ts">
import { ref, onMounted, onBeforeUnmount, watch } from 'vue'
import uPlot from 'uplot'
import 'uplot/dist/uPlot.min.css'
import { useDarkMode } from '@/composables/useDarkMode'

export interface AxisColors {
    axis: string
    grid: string
}

// A uPlot chart that rebuilds when its data or the theme changes and follows
// its container's width. `options` receives the theme's axis colours.
const props = defineProps<{
    data: uPlot.AlignedData
    options: (colors: AxisColors) => Omit<uPlot.Options, 'width'>
}>()

const { isDark } = useDarkMode()
const containerRef = ref<HTMLDivElement>()
let chart: uPlot | null = null

function createChart(): void {
    if (!containerRef.value) return
    chart?.destroy()
    const colors = isDark.value
        ? { axis: '#a1a1aa', grid: '#27272a' }
        : { axis: '#888', grid: '#eee' }
    chart = new uPlot(
        { width: containerRef.value.clientWidth || 800, ...props.options(colors) },
        props.data,
        containerRef.value,
    )
}

let resizeObserver: ResizeObserver | null = null

onMounted(() => {
    createChart()
    resizeObserver = new ResizeObserver(() => {
        if (chart && containerRef.value) {
            chart.setSize({ width: containerRef.value.clientWidth, height: chart.height })
        }
    })
    if (containerRef.value) resizeObserver.observe(containerRef.value)
})

onBeforeUnmount(() => {
    chart?.destroy()
    chart = null
    resizeObserver?.disconnect()
})

watch(() => [props.data, props.options, isDark.value], createChart)
</script>

<style scoped>
.uplot-chart {
    width: 100%;
}
</style>
