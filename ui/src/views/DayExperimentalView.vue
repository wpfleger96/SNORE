<template>
    <div v-if="loading" class="loading-state">
        <Loader2 class="inline h-4 w-4 animate-spin" /> Loading experimental metrics...
    </div>

    <ErrorState v-else-if="error" :message="error" :retry="reload" />

    <div v-else-if="!data" class="empty-day">
        <p class="empty-message">No CPAP data recorded for this date.</p>
        <RouterLink :to="`/days/${props.dayDate}`" class="back-link">
            <ArrowLeft class="inline h-4 w-4" /> Day detail
        </RouterLink>
    </div>

    <div v-else class="day-experimental">
        <RouterLink :to="`/days/${props.dayDate}`" class="back-link">
            <ArrowLeft class="inline h-4 w-4" /> Day detail
        </RouterLink>

        <h1 class="page-title">Experimental Metrics: {{ formatDateWithWeekday(data.date) }}</h1>

        <div class="space-y-3">
            <ExperimentalBanner
                body="SNORE's own flow-limitation and RERA-proxy analysis: a night-to-night trend instrument, not a clinical measurement."
            />
            <div class="stats-grid">
                <StatCard
                    label="FL Class ≥4"
                    :value="data.fl_class_ge4_pct ?? null"
                    :reason="data.fl_class_ge4_pct_reason"
                    unit="%"
                    :decimals="1"
                    glossary-key="fl_class_ge4_pct"
                />
                <StatCard
                    label="RERA Proxy Index"
                    :value="data.rera_index ?? null"
                    :reason="data.rera_index_reason"
                    :decimals="2"
                    glossary-key="rera_index"
                />
                <StatCard
                    label="RERA Proxy Count"
                    :value="data.rera_count ?? null"
                    :reason="data.rera_count_reason"
                    :decimals="0"
                    glossary-key="rera_count"
                />
            </div>
        </div>
    </div>
</template>

<script setup lang="ts">
import { watch } from 'vue'
import { isAxiosError } from 'axios'
import { ArrowLeft, Loader2 } from '@lucide/vue'
import ErrorState from '@/components/ErrorState.vue'
import ExperimentalBanner from '@/components/ExperimentalBanner.vue'
import StatCard from '@/components/StatCard.vue'
import { getDayExperimental } from '@/api/days'
import { useApiLoad } from '@/composables/useApiLoad'
import { formatDateWithWeekday } from '@/utils/formatting'

const props = defineProps<{ dayDate: string }>()

// A 404 is the "no such day" empty state, not a load failure.
const { data, loading, error, reload } = useApiLoad(async () => {
    try {
        return await getDayExperimental(props.dayDate)
    } catch (err) {
        if (isAxiosError(err) && err.response?.status === 404) return null
        throw err
    }
}, 'Failed to load experimental metrics')

// The router reuses this component across dates.
watch(() => props.dayDate, reload)
</script>

<style scoped>
.day-experimental {
    max-width: 1000px;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 0.75rem;
}

.empty-day {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    gap: 1rem;
    padding: 2rem 0;
}

.empty-message {
    color: var(--color-muted-foreground);
}
</style>
