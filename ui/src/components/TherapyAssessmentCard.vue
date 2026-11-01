<template>
    <div class="section-card space-y-3">
        <h2>Therapy Assessment</h2>
        <ExperimentalBanner
            body="SNORE's own judgement from AHI across all recorded data, not a clinical assessment."
        />
        <ErrorState v-if="error" :message="error" :retry="reload" />
        <div v-else-if="effectiveness || trendDirection" class="assessment-row">
            <div v-if="effectiveness" class="assessment-item">
                <span class="assessment-label"
                    >Effectiveness <InfoHint glossary-key="effectiveness"
                /></span>
                <Badge v-bind="effectivenessBadgeAttrs(effectiveness)" class="capitalize">
                    {{ effectiveness }}
                </Badge>
            </div>
            <div v-if="trendDirection" class="assessment-item">
                <span class="assessment-label"
                    >AHI Trend <InfoHint glossary-key="ahi_trend"
                /></span>
                <span class="trend-badge" :class="'trend-' + trendDirection">
                    {{ trendDirection }}
                </span>
            </div>
        </div>
        <p v-else-if="!loading" class="assessment-empty">
            Not enough therapy data for an assessment.
        </p>
    </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Badge } from '@/components/ui/badge'
import ErrorState from '@/components/ErrorState.vue'
import ExperimentalBanner from '@/components/ExperimentalBanner.vue'
import InfoHint from '@/components/InfoHint.vue'
import { getExperimentalStats } from '@/api/stats'
import { useApiLoad } from '@/composables/useApiLoad'
import type { ExperimentalStats } from '@/types'

const { data, loading, error, reload } = useApiLoad(
    () => getExperimentalStats(),
    'Failed to load therapy assessment',
)

type Effectiveness = Exclude<ExperimentalStats['effectiveness'], 'unknown'>

const effectiveness = computed<Effectiveness | null>(() => {
    const band = data.value?.effectiveness
    return band && band !== 'unknown' ? band : null
})
const trendDirection = computed(() => data.value?.ahi_trend_direction ?? null)

function effectivenessBadgeAttrs(e: Effectiveness): Record<string, string> {
    if (e === 'excellent' || e === 'good')
        return { class: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' }
    if (e === 'fair')
        return { class: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' }
    return { variant: 'destructive' }
}
</script>

<style scoped>
.assessment-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.75rem 2rem;
}

.assessment-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.assessment-label,
.assessment-empty {
    font-size: 0.875rem;
    color: var(--color-muted-foreground);
}

.trend-badge {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: capitalize;
    padding: 0.1rem 0.35rem;
    border-radius: 0.25rem;
    border: 1px solid currentColor;
}

.trend-improving {
    color: var(--color-success);
}

.trend-worsening {
    color: var(--color-destructive);
}

.trend-stable {
    color: var(--color-muted-foreground);
}
</style>
