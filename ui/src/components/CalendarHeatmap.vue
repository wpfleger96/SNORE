<template>
    <div ref="heatmapEl" class="calendar-heatmap">
        <div class="day-labels">
            <span>Mon</span>
            <span />
            <span>Wed</span>
            <span />
            <span>Fri</span>
            <span />
            <span />
        </div>
        <div>
            <div class="month-labels" :style="{ gridTemplateColumns: `repeat(${weeks}, 14px)` }">
                <span v-for="m in monthLabels" :key="m.offset" :style="{ gridColumn: m.offset }">
                    {{ m.label }}
                </span>
            </div>
            <div class="grid" :style="{ gridTemplateColumns: `repeat(${weeks}, 14px)` }">
                <div
                    v-for="cell in cells"
                    :key="cell.date"
                    class="cell"
                    :class="cell.class"
                    :title="cell.title"
                    @click="cell.ahi != null && $emit('day-click', cell.date)"
                />
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, nextTick, ref, watch } from 'vue'
import type { DayListItem } from '@/types'
import { parseLocalDate } from '@/utils/formatting'
import { ahiColorClass } from '@/utils/ahiScale'

const props = defineProps<{
    days: DayListItem[]
    monthsBack?: number
}>()

defineEmits<{
    'day-click': [date: string]
}>()

// Which value a day's headline AHI is (DayListItem.index_source), shown in the
// cell tooltip so a recount is never mistaken for the device's own number.
const AHI_SOURCE_LABELS: Record<NonNullable<DayListItem['index_source']>, string> = {
    device: 'device-reported',
    derived: 'SNORE recount',
}

function cellTitle(date: string, day: DayListItem | undefined): string {
    if (day?.ahi == null) return `${date}: AHI N/A`
    const source = day.index_source ? ` (${AHI_SOURCE_LABELS[day.index_source]})` : ''
    return `${date}: AHI ${day.ahi.toFixed(1)}${source}`
}

const monthsBack = computed(() => props.monthsBack ?? 6)

const heatmapEl = ref<HTMLElement | null>(null)

const dayMap = computed(() => {
    const map = new Map<string, DayListItem>()
    for (const d of props.days) map.set(d.date, d)
    return map
})

const cells = computed(() => {
    const end = new Date()
    const start = new Date()
    start.setMonth(start.getMonth() - monthsBack.value)
    // Align start to Monday
    const dayOfWeek = start.getDay()
    const diff = dayOfWeek === 0 ? -6 : 1 - dayOfWeek
    start.setDate(start.getDate() + diff)

    const result: { date: string; ahi: number | null; class: string; title: string }[] = []
    const cur = new Date(start)
    while (cur <= end) {
        const iso = cur.toISOString().slice(0, 10)
        const day = dayMap.value.get(iso)
        result.push({
            date: iso,
            ahi: day?.ahi ?? null,
            class: ahiColorClass(day?.ahi ?? null),
            title: cellTitle(iso, day),
        })
        cur.setDate(cur.getDate() + 1)
    }
    return result
})

const weeks = computed(() => Math.ceil(cells.value.length / 7))

// Show the most recent weeks first when the grid overflows horizontally.
// Watch the cells source (not just mount): monthsBack flips 3↔6 on rotation,
// rebuilding the grid, so the scroll anchor must be re-applied.
watch(
    cells,
    async () => {
        await nextTick()
        if (heatmapEl.value) heatmapEl.value.scrollLeft = heatmapEl.value.scrollWidth
    },
    { immediate: true },
)

// Week columns a month label needs before the next one (labels overflow their 14px column).
const MIN_LABEL_GAP = 3

const monthLabels = computed(() => {
    const labels: { label: string; offset: number }[] = []
    let lastMonth = -1
    for (let i = 0; i < cells.value.length; i += 7) {
        const d = parseLocalDate(cells.value[i].date)
        if (d.getMonth() !== lastMonth) {
            lastMonth = d.getMonth()
            const offset = Math.floor(i / 7) + 1
            // Monday alignment can start the grid in the previous month; drop that
            // spillover label rather than let it collide with the real month's.
            const prev = labels.at(-1)
            if (prev && offset - prev.offset < MIN_LABEL_GAP) labels.pop()
            labels.push({
                label: d.toLocaleString(undefined, { month: 'short' }),
                offset,
            })
        }
    }
    return labels
})
</script>

<style scoped>
.calendar-heatmap {
    display: flex;
    gap: 0.25rem;
    overflow-x: auto;
}

.day-labels {
    /* Pin to the left edge of the scroll container; cells slide under it. */
    position: sticky;
    left: 0;
    z-index: 1;
    background: var(--color-card);
    display: grid;
    grid-template-rows: repeat(7, 14px);
    gap: 2px;
    font-size: 0.65rem;
    color: var(--color-muted-foreground);
    /* Clear the month-label row (height + margin-bottom) so rows align with cells.
       Padding, not margin, so the sticky background also hides month labels
       scrolling past on mobile. */
    padding-top: 1.25rem;
    text-align: right;
    padding-right: 0.25rem;
}

.month-labels {
    display: grid;
    gap: 2px;
    font-size: 0.65rem;
    color: var(--color-muted-foreground);
    height: 1rem;
    line-height: 1rem;
    margin-bottom: 0.25rem;
    /* Labels are wider than a week column: overflow it rather than wrap. */
    white-space: nowrap;
}

.grid {
    display: grid;
    grid-template-rows: repeat(7, 14px);
    grid-auto-flow: column;
    gap: 2px;
}

.cell {
    width: 14px;
    height: 14px;
    border-radius: 2px;
    cursor: pointer;
}

.cell--empty {
    background: var(--color-muted);
    cursor: default;
}
.cell--good {
    background: #22c55e;
}
.cell--mild {
    background: #eab308;
}
.cell--moderate {
    background: #f97316;
}
.cell--severe {
    background: #ef4444;
}
</style>
