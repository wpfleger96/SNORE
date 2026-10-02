<template>
    <div v-if="loading" class="loading-state">
        <Loader2 class="inline h-4 w-4 animate-spin" /> Loading sleep data...
    </div>

    <ErrorState v-else-if="error" :message="error" :retry="loadData" />

    <div v-else-if="notFound" class="empty-night">
        <p class="empty-message">No Apple Health data for this night.</p>
        <RouterLink to="/apple-health" class="back-link">
            <ArrowLeft class="inline h-4 w-4" /> Back to Apple Health
        </RouterLink>
    </div>

    <div v-else-if="night" class="apple-health-night">
        <RouterLink to="/apple-health" class="back-link">
            <ArrowLeft class="inline h-4 w-4" /> Apple Health
        </RouterLink>

        <div class="night-header">
            <h1>{{ formatDateWithWeekday(night.night_date) }}</h1>
            <p v-if="night.preferred_source" class="source-attr">
                Source: {{ night.preferred_source }}
            </p>
        </div>

        <!-- Primary sleep stats -->
        <div class="stats-grid mb-6">
            <StatCard
                label="Time in Bed"
                :value="secToHours(night.time_in_bed_seconds)"
                unit="hr"
                :decimals="1"
                glossary-key="time_in_bed"
            />
            <StatCard
                label="Total Sleep"
                :value="secToHours(night.total_sleep_seconds)"
                unit="hr"
                :decimals="1"
                glossary-key="total_sleep"
            />
            <StatCard
                label="Efficiency"
                :value="night.sleep_efficiency_pct ?? null"
                unit="%"
                :decimals="1"
                glossary-key="sleep_efficiency"
            />
            <StatCard
                label="Core"
                :value="secToHours(night.core_seconds)"
                unit="hr"
                :decimals="1"
                glossary-key="core_sleep"
            />
            <StatCard
                label="Deep"
                :value="secToHours(night.deep_seconds)"
                unit="hr"
                :decimals="1"
                glossary-key="deep_sleep"
            />
            <StatCard
                label="REM"
                :value="secToHours(night.rem_seconds)"
                unit="hr"
                :decimals="1"
                glossary-key="rem_sleep"
            />
            <StatCard
                label="Awake"
                :value="secToHours(night.awake_seconds)"
                unit="hr"
                :decimals="1"
                glossary-key="awake_time"
            />
            <StatCard
                label="Stage Coverage"
                :value="night.stage_coverage_pct ?? null"
                unit="%"
                :decimals="1"
            />
        </div>

        <!-- Oximetry + respiratory rate (only when data is present) -->
        <div
            v-if="night.avg_spo2_pct != null || night.min_spo2_pct != null || night.avg_rr != null"
            class="stats-grid mb-6"
        >
            <StatCard
                v-if="night.avg_spo2_pct != null"
                label="SpO₂ Avg"
                :value="night.avg_spo2_pct"
                unit="%"
                :decimals="1"
                glossary-key="spo2"
            />
            <StatCard
                v-if="night.min_spo2_pct != null"
                label="SpO₂ Min"
                :value="night.min_spo2_pct"
                unit="%"
                :decimals="1"
                glossary-key="spo2"
            />
            <StatCard
                v-if="night.avg_rr != null"
                label="Resp Rate"
                :value="night.avg_rr"
                unit="br/min"
                :decimals="1"
                glossary-key="resp_rate"
            />
        </div>

        <!-- Sleep stage hypnogram -->
        <div v-if="samples.length" class="chart-section">
            <h2 class="section-heading">Sleep Stages</h2>
            <SleepStageChart :samples="samples" :height="220" />
        </div>

        <div class="footer-links">
            <RouterLink :to="`/days/${props.nightDate}`" class="cross-link">
                View CPAP day
                <ArrowRight class="inline h-4 w-4" />
            </RouterLink>
        </div>
    </div>
</template>

<script setup lang="ts">
import { onMounted, ref, shallowRef, watch } from 'vue'
import { isAxiosError } from 'axios'
import { ArrowLeft, ArrowRight, Loader2 } from '@lucide/vue'
import StatCard from '@/components/StatCard.vue'
import ErrorState from '@/components/ErrorState.vue'
import SleepStageChart from '@/components/SleepStageChart.vue'
import { getHealthNight, getHealthNightSamples } from '@/api/health'
import { formatDateWithWeekday, secToHours } from '@/utils/formatting'
import type { HealthNightDetailRead, HealthSampleRead } from '@/types'

const props = defineProps<{ nightDate: string }>()

const loading = ref(true)
const error = ref<string | null>(null)
const notFound = ref(false)
const night = shallowRef<HealthNightDetailRead | null>(null)
const samples = shallowRef<HealthSampleRead[]>([])

async function loadData(): Promise<void> {
    loading.value = true
    error.value = null
    notFound.value = false
    try {
        const [nightData, samplesData] = await Promise.all([
            getHealthNight(props.nightDate),
            getHealthNightSamples(props.nightDate),
        ])
        night.value = nightData
        samples.value = samplesData
    } catch (err) {
        if (isAxiosError(err) && err.response?.status === 404) {
            notFound.value = true
        } else {
            error.value = err instanceof Error ? err.message : 'Failed to load sleep data'
        }
    } finally {
        loading.value = false
    }
}

onMounted(() => void loadData())
watch(
    () => props.nightDate,
    () => void loadData(),
)
</script>

<style scoped>
.apple-health-night {
    max-width: 1000px;
}

.night-header {
    margin-bottom: 1.5rem;
}

.night-header h1 {
    font-size: 1.5rem;
    font-weight: 600;
}

.source-attr {
    font-size: 0.875rem;
    color: var(--color-muted-foreground);
    margin-top: 0.25rem;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 0.75rem;
}

.mb-6 {
    margin-bottom: 1.5rem;
}

.chart-section {
    margin-bottom: 1.5rem;
}

.section-heading {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
}

.footer-links {
    margin-top: 1.5rem;
    display: flex;
    gap: 1.5rem;
}

.cross-link {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.875rem;
    color: var(--color-primary);
    text-decoration: none;
}

.cross-link:hover {
    text-decoration: underline;
}

.empty-night {
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
