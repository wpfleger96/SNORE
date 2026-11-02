<template>
    <div class="stats-view">
        <h1 class="page-title">Statistics</h1>

        <!-- Controls -->
        <div class="controls">
            <div class="control-group">
                <span class="control-label">Granularity</span>
                <ToggleGroup
                    :model-value="granularity"
                    type="single"
                    variant="outline"
                    @update:model-value="
                        (v) => {
                            if (v) setGranularity(v as string)
                        }
                    "
                >
                    <ToggleGroupItem
                        v-for="opt in granularityOptions"
                        :key="opt.value"
                        :value="opt.value"
                    >
                        {{ opt.label }}
                    </ToggleGroupItem>
                </ToggleGroup>
            </div>

            <div class="control-group">
                <span class="control-label">Range</span>
                <ToggleGroup
                    :model-value="daysRange"
                    type="single"
                    variant="outline"
                    @update:model-value="
                        (v) => {
                            if (v) daysRange = v as string
                        }
                    "
                >
                    <ToggleGroupItem
                        v-for="opt in rangeOptions"
                        :key="opt.value"
                        :value="opt.value"
                        :disabled="opt.value === 'all' && granularity === 'day'"
                    >
                        {{ opt.label }}
                    </ToggleGroupItem>
                </ToggleGroup>
            </div>

            <div class="control-group">
                <span class="control-label">Metrics</span>
                <ToggleGroup
                    :model-value="selectedMetrics"
                    type="multiple"
                    variant="outline"
                    class="metrics-toggle"
                    @update:model-value="(v) => setSelectedMetrics(v as string[])"
                >
                    <ToggleGroupItem v-for="key in metricKeys" :key="key" :value="key">
                        {{ METRIC_CONFIG[key].label }}
                    </ToggleGroupItem>
                </ToggleGroup>
            </div>
        </div>

        <!-- Period Stats Table -->
        <div class="section-card">
            <h2>Period Breakdown</h2>
            <ErrorState v-if="periodsError" :message="periodsError" :retry="reloadPeriods" />
            <PeriodStatsTable
                v-else
                :periods="periods"
                :loading="periodsLoading || dataRangeLoading"
                :empty-message="periodsEmptyMessage"
                :show-sleep-columns="hasSleepPeriodData"
            />
        </div>

        <!-- Trend Charts (one per selected metric) -->
        <div v-if="!periodsError && trendLabels.length" class="section-card">
            <h2>Trends</h2>
            <template v-for="key in selectedMetrics" :key="key">
                <div v-if="hasData(key)" class="trend-metric">
                    <p class="trend-metric-label">
                        {{ METRIC_CONFIG[key].label }}
                        <InfoHint :glossary-key="METRIC_CONFIG[key].glossaryKey" />
                    </p>
                    <TrendChart
                        :labels="trendLabels"
                        :datasets="[metricDataset(key)]"
                        :height="200"
                        :sync-key="trendSync"
                    />
                </div>
            </template>
            <p v-if="!anyVisibleChart" class="no-data-hint">
                No data available for the selected metrics.
            </p>
        </div>

        <!-- Records -->
        <div class="section-card">
            <h2>Records</h2>
            <ErrorState v-if="recordsError" :message="recordsError" :retry="reloadRecords" />
            <RecordsPanel
                v-else
                :records="records"
                :loading="recordsLoading || dataRangeLoading"
                :empty-message="recordsEmptyMessage"
            />
        </div>
    </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import uPlot from 'uplot'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import PeriodStatsTable from '@/components/PeriodStatsTable.vue'
import TrendChart from '@/components/TrendChart.vue'
import RecordsPanel from '@/components/RecordsPanel.vue'
import ErrorState from '@/components/ErrorState.vue'
import InfoHint from '@/components/InfoHint.vue'
import { getPeriods, getTrends, getRecords, getDataRange } from '@/api/stats'
import { useApiLoad } from '@/composables/useApiLoad'
import { formatDateFull } from '@/utils/formatting'
import { GLOSSARY } from '@/utils/glossary'
import type { PeriodStatistics, TrendData } from '@/types'

// ────────────────────────────── Metric config ──────────────────────────────

// glossaryKey is the GLOSSARY key for InfoHint; differs from the METRIC_CONFIG entry key only
// for rr (glossary: resp_rate). Labels are composed from GLOSSARY where the label matches the
// desired display name; rr/mv/rera keep local names because their glossary labels differ.
const METRIC_CONFIG: Record<
    string,
    { label: string; glossaryKey: string; key: keyof TrendData; color: string }
> = {
    ahi: {
        label: `${GLOSSARY.ahi.label} (events/hr)`,
        glossaryKey: 'ahi',
        key: 'ahi',
        color: '#2563eb',
    },
    usage: {
        label: `${GLOSSARY.usage.label} (hrs)`,
        glossaryKey: 'usage',
        key: 'usage',
        color: '#16a34a',
    },
    spo2: {
        label: `${GLOSSARY.spo2.label} (%)`,
        glossaryKey: 'spo2',
        key: 'spo2',
        color: '#f97316',
    },
    leak: {
        label: `${GLOSSARY.leak.label} (L/min)`,
        glossaryKey: 'leak',
        key: 'leak',
        color: '#dc2626',
    },
    pressure: {
        label: `${GLOSSARY.pressure.label} (cmH₂O)`,
        glossaryKey: 'pressure',
        key: 'pressure',
        color: '#7c3aed',
    },
    epap: {
        label: `${GLOSSARY.epap.label} (cmH₂O)`,
        glossaryKey: 'epap',
        key: 'epap',
        color: '#06b6d4',
    },
    rr: { label: 'Resp Rate (br/min)', glossaryKey: 'resp_rate', key: 'rr', color: '#db2777' },
    pulse: {
        label: `${GLOSSARY.pulse.label} (BPM)`,
        glossaryKey: 'pulse',
        key: 'pulse',
        color: '#d97706',
    },
    mv: { label: 'Minute Vent (L/min)', glossaryKey: 'mv', key: 'mv', color: '#059669' },
    oai: {
        label: `${GLOSSARY.oai.label} (events/hr)`,
        glossaryKey: 'oai',
        key: 'oai',
        color: '#be123c',
    },
    cai: {
        label: `${GLOSSARY.cai.label} (events/hr)`,
        glossaryKey: 'cai',
        key: 'cai',
        color: '#0284c7',
    },
    hi: {
        label: `${GLOSSARY.hi.label} (events/hr)`,
        glossaryKey: 'hi',
        key: 'hi',
        color: '#ca8a04',
    },
    rera: { label: 'RERA (events/hr)', glossaryKey: 'rera', key: 'rera', color: '#ea580c' },
    total_sleep_hours: {
        label: 'Avg Sleep (hrs)',
        glossaryKey: 'total_sleep',
        key: 'total_sleep_hours',
        color: '#4f46e5',
    },
    sleep_efficiency: {
        label: 'Sleep Efficiency (%)',
        glossaryKey: 'sleep_efficiency',
        key: 'sleep_efficiency',
        color: '#8b5cf6',
    },
}

const VALID_METRICS = new Set(Object.keys(METRIC_CONFIG))
const DEFAULT_METRICS = ['ahi', 'usage', 'spo2', 'leak']
const STORAGE_KEY = 'snore:trend-metrics'

// ────────────────────────────── Options ──────────────────────────────

const granularityOptions = [
    { label: 'Day', value: 'day' },
    { label: 'Week', value: 'week' },
    { label: 'Month', value: 'month' },
    { label: '6 Month', value: '6month' },
    { label: 'Year', value: 'year' },
]

// Toggle button labels ('30d', '1yr', 'All') differ from the message labels ('30 days', '1 year',
// 'all time'), so rangeOptions is kept as a separate structure rather than derived from RANGE_CONFIG.
const rangeOptions = [
    { label: '30d', value: '30d' },
    { label: '90d', value: '90d' },
    { label: '180d', value: '180d' },
    { label: '1yr', value: '1yr' },
    { label: 'All', value: 'all' },
]

const RANGE_CONFIG: Record<string, { days?: number; label: string }> = {
    '30d': { days: 30, label: '30 days' },
    '90d': { days: 90, label: '90 days' },
    '180d': { days: 180, label: '180 days' },
    '1yr': { days: 365, label: '1 year' },
    all: { days: undefined, label: 'all time' },
}

// ────────────────────────────── State ──────────────────────────────

const granularity = ref('month')
const daysRange = ref('90d')

// When switching to day granularity, deselect 'All' to avoid unbounded day-level fetch.
// 'All' is also visually disabled in the Range toggle while day is active.
function setGranularity(v: string): void {
    granularity.value = v
    if (v === 'day' && daysRange.value === 'all') {
        daysRange.value = '180d'
    }
}

const effectiveDaysLimit = computed<number | undefined>(() => RANGE_CONFIG[daysRange.value]?.days)

// ────────────────────────────── Metric selection + persistence ──────────────────────────────

function loadStoredMetrics(): string[] {
    try {
        const raw = localStorage.getItem(STORAGE_KEY)
        if (!raw) return DEFAULT_METRICS
        const parsed: unknown = JSON.parse(raw)
        if (!Array.isArray(parsed)) return DEFAULT_METRICS
        const valid = parsed.filter(
            (k): k is string => typeof k === 'string' && VALID_METRICS.has(k),
        )
        return valid.length ? valid : DEFAULT_METRICS
    } catch {
        return DEFAULT_METRICS
    }
}

const selectedMetrics = ref<string[]>(loadStoredMetrics())

function setSelectedMetrics(v: string[]): void {
    if (v.length === 0) return
    selectedMetrics.value = v
    localStorage.setItem(STORAGE_KEY, JSON.stringify(v))
}

// ────────────────────────────── Data fetching ──────────────────────────────

const trendSync = uPlot.sync('trend-charts')

const {
    data: periodData,
    loading: periodsLoading,
    error: periodsError,
    reload: reloadPeriods,
} = useApiLoad(async () => {
    const [periods, trends] = await Promise.all([
        getPeriods(granularity.value, effectiveDaysLimit.value),
        getTrends(granularity.value, effectiveDaysLimit.value),
    ])
    return { periods, trends }
})

const {
    data: records,
    loading: recordsLoading,
    error: recordsError,
    reload: reloadRecords,
} = useApiLoad(() => getRecords(effectiveDaysLimit.value))

const periods = computed<PeriodStatistics[]>(() => periodData.value?.periods ?? [])
const trends = computed(() => periodData.value?.trends ?? null)
const hasSleepPeriodData = computed(() =>
    periods.value.some((p) => p.avg_total_sleep_hours != null),
)

watch([granularity, daysRange], () => {
    void reloadPeriods()
    void reloadRecords()
})

// Mount-only; the all-time latest data date doesn't change with the range picker.
// Errors are intentionally not surfaced — on failure, empty states fall back to generic copy.
const { data: dataRange, loading: dataRangeLoading } = useApiLoad(() => getDataRange())

// ────────────────────────────── Empty-state messages ──────────────────────────────

function rangeEmptyMessage(noun: string, fallback: string): string {
    if (dataRange.value?.latest_date && daysRange.value !== 'all') {
        const { label } = RANGE_CONFIG[daysRange.value]
        const formattedDate = formatDateFull(dataRange.value.latest_date)
        return `No ${noun} in the last ${label} — most recent night is ${formattedDate}. Try a wider range.`
    }
    return fallback
}

const periodsEmptyMessage = computed(() => rangeEmptyMessage('data', 'No period data available.'))
const recordsEmptyMessage = computed(() => rangeEmptyMessage('records', 'No records available.'))

// ────────────────────────────── Chart helpers ──────────────────────────────

const trendLabels = computed(() => trends.value?.ahi?.map((t) => t[0]) ?? [])

function hasData(key: string): boolean {
    const cfg = METRIC_CONFIG[key]
    if (!cfg || !trends.value) return false
    const series = trends.value[cfg.key]
    return Array.isArray(series) && series.length > 0 && series.some((t) => t[1] !== null)
}

function metricDataset(key: string): { label: string; values: (number | null)[]; color: string } {
    const cfg = METRIC_CONFIG[key]
    const series = trends.value?.[cfg.key] ?? []
    return {
        label: cfg.label,
        values: series.map((t) => t[1]),
        color: cfg.color,
    }
}

const anyVisibleChart = computed(() => selectedMetrics.value.some((key) => hasData(key)))

// An all-null RERA trend means no device in range scores RERAs, so offer no toggle for it. The
// key stays in selectedMetrics, so the chart returns if a wider range brings RERA data back.
function isOffered(key: string): boolean {
    if (key !== 'rera' || !trends.value) return true
    return hasData(key)
}

const metricKeys = computed(() => Object.keys(METRIC_CONFIG).filter(isOffered))
</script>

<style scoped>
.stats-view {
    max-width: 1200px;
}

.controls {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    margin-bottom: 1.25rem;
}

.control-group {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
}

.control-label {
    flex-shrink: 0;
    width: 6rem;
    padding-top: 0.5rem;
    font-size: 0.875rem;
    color: var(--muted-foreground);
}

.metrics-toggle {
    flex-wrap: wrap;
}

.trend-metric {
    margin-bottom: 1rem;
}

.trend-metric:last-child {
    margin-bottom: 0;
}

.trend-metric-label {
    font-size: 0.8125rem;
    font-weight: 500;
    color: var(--muted-foreground);
    margin-bottom: 0.25rem;
}

.no-data-hint {
    font-size: 0.875rem;
    color: var(--muted-foreground);
    text-align: center;
    padding: 1.5rem 0;
}

@media (max-width: 767.98px) {
    .control-group {
        flex-direction: column;
        align-items: stretch;
        gap: 0.375rem;
    }

    .control-label {
        width: auto;
        padding-top: 0;
    }

    .control-group [data-slot='toggle-group'] {
        flex-wrap: wrap;
    }
}
</style>
