<template>
    <div class="data-card">
        <div class="data-card-header">
            <RouterLink
                :to="`/apple-health/${night.night_date}`"
                class="text-primary no-underline hover:underline"
            >
                {{ formatDateFull(night.night_date) }}
            </RouterLink>
        </div>
        <div v-for="row in rows" :key="row.field" class="data-card-row">
            <span class="data-card-label"
                >{{ row.label }} <ProvenanceMark :provenance="provenanceFor(row.field)"
            /></span>
            <span class="data-card-value">{{ row.value }}</span>
        </div>
        <div class="data-card-row">
            <span class="data-card-label">Source</span>
            <span class="data-card-value">{{ night.preferred_source ?? '---' }}</span>
        </div>
    </div>
</template>

<script setup lang="ts">
// Mobile card for one Apple Health night. Only the date is a link, so the
// provenance marks (buttons) are never nested inside it.
import { computed } from 'vue'
import ProvenanceMark from '@/components/ProvenanceMark.vue'
import type { HealthNightSummaryRead } from '@/types'
import { formatDateFull } from '@/utils/formatting'
import { provenanceFor, type ProvenanceKey } from '@/utils/provenance'

const props = defineProps<{ night: HealthNightSummaryRead }>()

function fmtHours(seconds: number | null | undefined): string {
    return seconds != null ? (seconds / 3600).toFixed(1) : '---'
}

const rows = computed((): { label: string; field: ProvenanceKey; value: string }[] => {
    const n = props.night
    return [
        {
            label: 'Total Sleep (hr)',
            field: 'HealthNightSummaryRead.total_sleep_seconds',
            value: fmtHours(n.total_sleep_seconds),
        },
        {
            label: 'Efficiency (%)',
            field: 'HealthNightSummaryRead.sleep_efficiency_pct',
            value: n.sleep_efficiency_pct != null ? n.sleep_efficiency_pct.toFixed(1) : '---',
        },
        {
            label: 'Core (hr)',
            field: 'HealthNightSummaryRead.core_seconds',
            value: fmtHours(n.core_seconds),
        },
        {
            label: 'Deep (hr)',
            field: 'HealthNightSummaryRead.deep_seconds',
            value: fmtHours(n.deep_seconds),
        },
        {
            label: 'REM (hr)',
            field: 'HealthNightSummaryRead.rem_seconds',
            value: fmtHours(n.rem_seconds),
        },
    ]
})
</script>
