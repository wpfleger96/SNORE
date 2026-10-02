<template>
    <div v-if="loading" class="records-placeholder">
        <Loader2 class="h-4 w-4 animate-spin" /> Loading records...
    </div>
    <div v-else-if="!displayMetrics.length" class="records-placeholder">
        {{ props.emptyMessage ?? 'No records available.' }}
    </div>
    <div v-else class="records-grid">
        <div v-for="(metric, key) in displayMetrics" :key="key" class="record-card">
            <h4>{{ metric.label }} <InfoHint :glossary-key="metric.glossaryKey" /></h4>
            <div class="record-columns">
                <div class="record-col">
                    <span class="col-header best-header">Best</span>
                    <div v-for="(entry, i) in metric.best" :key="'b' + i" class="record-entry">
                        <span class="record-date">{{ formatDateMonthDay(entry[0]) }}</span>
                        <span class="record-value best-value">{{
                            entry[1].toFixed(metric.decimals)
                        }}</span>
                    </div>
                    <div v-if="!metric.best.length" class="record-empty">---</div>
                </div>
                <div class="record-col">
                    <span class="col-header worst-header">Worst</span>
                    <div v-for="(entry, i) in metric.worst" :key="'w' + i" class="record-entry">
                        <span class="record-date">{{ formatDateMonthDay(entry[0]) }}</span>
                        <span class="record-value worst-value">{{
                            entry[1].toFixed(metric.decimals)
                        }}</span>
                    </div>
                    <div v-if="!metric.worst.length" class="record-empty">---</div>
                </div>
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Loader2 } from '@lucide/vue'
import { formatDateMonthDay } from '@/utils/formatting'
import InfoHint from '@/components/InfoHint.vue'
import type { RecordsData } from '@/types'

const props = defineProps<{
    records: RecordsData | null
    loading: boolean
    emptyMessage?: string
}>()

const METRIC_CONFIG: Record<string, { label: string; decimals: number; glossaryKey: string }> = {
    ahi: { label: 'AHI', decimals: 1, glossaryKey: 'ahi' },
    leak: { label: 'Leak (L/min)', decimals: 1, glossaryKey: 'leak' },
    therapy_hours: { label: 'Therapy Hours', decimals: 1, glossaryKey: 'usage' },
    spo2_min: { label: 'SpO₂ Min (%)', decimals: 0, glossaryKey: 'spo2' },
    total_sleep_hours: { label: 'Total Sleep (hrs)', decimals: 1, glossaryKey: 'total_sleep' },
}

const displayMetrics = computed(() => {
    if (!props.records) return []
    return Object.entries(METRIC_CONFIG)
        .filter(([key]) => key in (props.records ?? {}))
        .map(([key, cfg]) => ({
            label: cfg.label,
            decimals: cfg.decimals,
            glossaryKey: cfg.glossaryKey,
            best: props.records![key]?.best ?? [],
            worst: props.records![key]?.worst ?? [],
        }))
})
</script>

<style scoped>
.records-placeholder {
    padding: 2rem;
    text-align: center;
    color: var(--color-muted-foreground);
}

.records-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
    gap: 1rem;
}

.record-card {
    background: var(--color-card);
    border: 1px solid var(--color-border);
    border-radius: 8px;
    padding: 1rem;
}

.record-card h4 {
    font-size: 0.85rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
    color: var(--color-foreground);
}

.record-columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
}

.col-header {
    display: block;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.4rem;
    font-weight: 600;
}

.best-header {
    color: var(--color-success);
}
.worst-header {
    color: var(--color-destructive);
}

.record-entry {
    display: flex;
    justify-content: space-between;
    font-size: 0.8rem;
    padding: 0.15rem 0;
}

.record-date {
    color: var(--color-muted-foreground);
}

.best-value {
    color: var(--color-success);
    font-weight: 600;
}
.worst-value {
    color: var(--color-destructive);
    font-weight: 600;
}

.record-empty {
    font-size: 0.8rem;
    color: var(--color-muted-foreground);
}
</style>
