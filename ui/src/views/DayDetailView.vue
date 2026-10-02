<template>
    <div v-if="loading" class="loading-state">
        <Loader2 class="inline h-4 w-4 animate-spin" /> Loading day...
    </div>

    <ErrorState v-else-if="error" :message="error" :retry="reload" />

    <div v-else-if="notFound" class="empty-day">
        <p class="empty-message">No CPAP data recorded for this date.</p>
        <RouterLink :to="`/apple-health/${props.dayDate}`" class="back-link">
            <ArrowLeft class="inline h-4 w-4" /> Apple Health night detail
        </RouterLink>
    </div>

    <div v-else-if="data" class="day-detail">
        <RouterLink to="/" class="back-link">
            <ArrowLeft class="inline h-4 w-4" /> Dashboard
        </RouterLink>

        <div class="day-header">
            <h1>{{ formatDateWithWeekday(data.date) }}</h1>
        </div>

        <div class="stats-grid mb-6">
            <StatCard
                label="Total Hours"
                :value="data.total_therapy_hours ?? null"
                unit="hr"
                :decimals="1"
                glossary-key="usage"
            />
            <StatCard label="AHI" :value="data.ahi ?? null" :decimals="1" glossary-key="ahi">
                <template v-if="ahiSourceNote" #default>{{ ahiSourceNote }}</template>
            </StatCard>
            <StatCard label="Sessions" :value="data.session_count" :decimals="0" />
            <StatCard label="OAI" :value="data.oai ?? null" :decimals="2" glossary-key="oai" />
            <StatCard label="CAI" :value="data.cai ?? null" :decimals="2" glossary-key="cai" />
            <StatCard label="HI" :value="data.hi ?? null" :decimals="2" glossary-key="hi" />
            <StatCard
                label="Obstructive Apneas"
                :value="data.obstructive_apneas"
                :decimals="0"
                glossary-key="obstructive_apneas"
            />
            <StatCard
                label="Central Apneas"
                :value="data.central_apneas"
                :decimals="0"
                glossary-key="central_apneas"
            />
            <StatCard
                label="Hypopneas"
                :value="data.hypopneas"
                :decimals="0"
                glossary-key="hypopneas"
            />
            <StatCard label="RERAs" :value="data.reras" :decimals="0" glossary-key="reras" />
        </div>

        <!-- SNORE breath-analysis group (experimental) -->
        <div v-if="hasBreathAnalysis" class="mb-6 space-y-3">
            <ExperimentalBanner
                body="SNORE's own flow-limitation and RERA-proxy analysis: a night-to-night trend instrument, not a clinical measurement."
            />
            <div class="stats-grid">
                <StatCard
                    v-if="data.fl_class_ge4_pct != null || data.fl_class_ge4_pct_reason != null"
                    label="FL Class ≥4"
                    :value="data.fl_class_ge4_pct ?? null"
                    :reason="data.fl_class_ge4_pct_reason"
                    unit="%"
                    :decimals="1"
                    glossary-key="fl_class_ge4_pct"
                />
                <StatCard
                    v-if="data.rera_index != null || data.rera_index_reason != null"
                    label="RERA Proxy Index"
                    :value="data.rera_index ?? null"
                    :reason="data.rera_index_reason"
                    :decimals="2"
                    glossary-key="rera_index"
                />
                <StatCard
                    v-if="data.rera_count != null || data.rera_count_reason != null"
                    label="RERA Proxy Count"
                    :value="data.rera_count ?? null"
                    :reason="data.rera_count_reason"
                    :decimals="0"
                    glossary-key="rera_count"
                />
            </div>
        </div>

        <!-- Pressure group -->
        <div
            v-if="
                data.avg_pressure != null || data.pressure_min != null || data.pressure_95th != null
            "
            class="stats-grid mb-6"
        >
            <StatCard
                v-if="data.avg_pressure != null"
                label="Pressure Mean"
                :value="data.avg_pressure"
                unit="cmH₂O"
                :decimals="1"
                glossary-key="pressure"
            />
            <StatCard
                v-if="data.pressure_min != null"
                label="Pressure Min"
                :value="data.pressure_min"
                unit="cmH₂O"
                :decimals="1"
            />
            <StatCard
                v-if="data.pressure_max != null"
                label="Pressure Max"
                :value="data.pressure_max"
                unit="cmH₂O"
                :decimals="1"
            />
            <StatCard
                v-if="data.pressure_median != null"
                label="Pressure Median"
                :value="data.pressure_median"
                unit="cmH₂O"
                :decimals="1"
            />
            <StatCard
                v-if="data.pressure_95th != null"
                label="Pressure 95th"
                :value="data.pressure_95th"
                unit="cmH₂O"
                :decimals="1"
            />
        </div>

        <!-- EPAP group -->
        <div
            v-if="data.epap_min != null || data.epap_mean != null || data.epap_95th != null"
            class="stats-grid mb-6"
        >
            <StatCard
                v-if="data.epap_mean != null"
                label="EPAP Mean"
                :value="data.epap_mean"
                unit="cmH₂O"
                :decimals="1"
                glossary-key="epap"
            />
            <StatCard
                v-if="data.epap_min != null"
                label="EPAP Min"
                :value="data.epap_min"
                unit="cmH₂O"
                :decimals="1"
            />
            <StatCard
                v-if="data.epap_max != null"
                label="EPAP Max"
                :value="data.epap_max"
                unit="cmH₂O"
                :decimals="1"
            />
            <StatCard
                v-if="data.epap_median != null"
                label="EPAP Median"
                :value="data.epap_median"
                unit="cmH₂O"
                :decimals="1"
            />
            <StatCard
                v-if="data.epap_95th != null"
                label="EPAP 95th"
                :value="data.epap_95th"
                unit="cmH₂O"
                :decimals="1"
            />
        </div>

        <!-- Leak group -->
        <div
            v-if="data.avg_leak != null || data.leak_min != null || data.leak_95th != null"
            class="stats-grid mb-6"
        >
            <StatCard
                v-if="data.avg_leak != null"
                label="Leak Mean"
                :value="data.avg_leak"
                unit="L/min"
                :decimals="1"
                glossary-key="leak"
            />
            <StatCard
                v-if="data.leak_min != null"
                label="Leak Min"
                :value="data.leak_min"
                unit="L/min"
                :decimals="1"
            />
            <StatCard
                v-if="data.leak_max != null"
                label="Leak Max"
                :value="data.leak_max"
                unit="L/min"
                :decimals="1"
            />
            <StatCard
                v-if="data.leak_95th != null"
                label="Leak 95th"
                :value="data.leak_95th"
                unit="L/min"
                :decimals="1"
            />
        </div>

        <!-- SpO₂ group -->
        <div
            v-if="data.avg_spo2 != null || data.spo2_min != null || data.spo2_max != null"
            class="stats-grid mb-6"
        >
            <StatCard
                v-if="data.avg_spo2 != null"
                label="SpO₂ Mean"
                :value="data.avg_spo2"
                unit="%"
                :decimals="1"
                glossary-key="spo2"
            />
            <StatCard
                v-if="data.spo2_min != null"
                label="SpO₂ Min"
                :value="data.spo2_min"
                unit="%"
                :decimals="1"
            />
            <StatCard
                v-if="data.spo2_max != null"
                label="SpO₂ Max"
                :value="data.spo2_max"
                unit="%"
                :decimals="1"
            />
        </div>

        <!-- Apple Health group -->
        <template v-if="data.health_sleep">
            <div class="apple-health-section">
                <h2>Apple Health</h2>
                <div class="stats-grid mb-4">
                    <StatCard
                        label="Time in Bed"
                        :value="secToHours(data.health_sleep.time_in_bed_seconds)"
                        unit="hr"
                        :decimals="1"
                        glossary-key="time_in_bed"
                    />
                    <StatCard
                        label="Total Sleep"
                        :value="secToHours(data.health_sleep.total_sleep_seconds)"
                        unit="hr"
                        :decimals="1"
                        glossary-key="total_sleep"
                    />
                    <StatCard
                        label="Efficiency"
                        :value="data.health_sleep.sleep_efficiency_pct ?? null"
                        unit="%"
                        :decimals="1"
                        glossary-key="sleep_efficiency"
                    />
                    <StatCard
                        label="Core"
                        :value="secToHours(data.health_sleep.core_seconds)"
                        unit="hr"
                        :decimals="1"
                        glossary-key="core_sleep"
                    />
                    <StatCard
                        label="Deep"
                        :value="secToHours(data.health_sleep.deep_seconds)"
                        unit="hr"
                        :decimals="1"
                        glossary-key="deep_sleep"
                    />
                    <StatCard
                        label="REM"
                        :value="secToHours(data.health_sleep.rem_seconds)"
                        unit="hr"
                        :decimals="1"
                        glossary-key="rem_sleep"
                    />
                </div>
                <RouterLink :to="`/apple-health/${data.date}`" class="session-link">
                    Apple Health night detail →
                </RouterLink>
            </div>
        </template>

        <div v-if="data.session_ids?.length" class="sessions-section">
            <h2>Sessions</h2>
            <Table>
                <TableHeader>
                    <TableRow>
                        <TableHead>#</TableHead>
                        <TableHead>Session ID</TableHead>
                    </TableRow>
                </TableHeader>
                <TableBody>
                    <TableRow v-for="(id, i) in data.session_ids" :key="id">
                        <TableCell class="text-muted-foreground">{{ i + 1 }}</TableCell>
                        <TableCell>
                            <RouterLink :to="`/sessions/${id}`" class="session-link">
                                {{ id }}
                            </RouterLink>
                        </TableCell>
                    </TableRow>
                </TableBody>
            </Table>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, shallowRef } from 'vue'
import { isAxiosError } from 'axios'
import StatCard from '@/components/StatCard.vue'
import ExperimentalBanner from '@/components/ExperimentalBanner.vue'
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table'
import { Loader2, ArrowLeft } from '@lucide/vue'
import { getDay } from '@/api/days'
import { formatDateWithWeekday, secToHours } from '@/utils/formatting'
import ErrorState from '@/components/ErrorState.vue'
import type { DayDetail } from '@/types'

const props = defineProps<{ dayDate: string }>()

const data = shallowRef<DayDetail | null>(null)
const loading = ref(true)
const error = ref<string | null>(null)
const notFound = ref(false)

async function reload(): Promise<void> {
    loading.value = true
    error.value = null
    notFound.value = false
    try {
        data.value = await getDay(props.dayDate)
    } catch (err) {
        if (isAxiosError(err) && err.response?.status === 404) {
            notFound.value = true
        } else {
            error.value = err instanceof Error ? err.message : 'Failed to load day'
        }
    } finally {
        loading.value = false
    }
}

// Which value the headline AHI is (index_source), so a recount is never
// mistaken for the device's own number.
const ahiSourceNote = computed((): string | null => {
    const day = data.value
    if (day?.ahi == null) return null
    if (day.index_source === 'derived') return 'SNORE recount'
    if (day.index_source !== 'device') return null
    return day.ahi_computed != null
        ? `Device-reported; SNORE recount ${day.ahi_computed.toFixed(1)}`
        : 'Device-reported'
})

const hasBreathAnalysis = computed(() => {
    const day = data.value
    return [
        day?.fl_class_ge4_pct,
        day?.fl_class_ge4_pct_reason,
        day?.rera_index,
        day?.rera_index_reason,
        day?.rera_count,
        day?.rera_count_reason,
    ].some((v) => v != null)
})

onMounted(() => void reload())
</script>

<style scoped>
.day-detail {
    max-width: 1000px;
}

.day-header {
    margin-bottom: 1.5rem;
}

.day-header h1 {
    font-size: 1.5rem;
    font-weight: 600;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 0.75rem;
}

.apple-health-section {
    margin-bottom: 1.5rem;
}

.apple-health-section h2 {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
}

.sessions-section h2 {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
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

.session-link {
    color: var(--color-primary);
}

.session-link:hover {
    text-decoration: underline;
}
</style>
