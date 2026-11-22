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
                        :datasets="metricDatasets(key)"
                        :height="200"
                        :sync-key="trendSync"
                        :stacked="METRIC_CONFIG[key].stacked"
                        :format-value="METRIC_CONFIG[key].formatValue"
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
import { formatDateFull, formatHoursAfterNoon } from '@/utils/formatting'
import { GLOSSARY } from '@/utils/glossary'
import type { PeriodStatistics, TrendData } from '@/types'

type PeriodSeriesKey = 'device_flg_95th' | 'snore_95th'
type SeriesKey = keyof TrendData | PeriodSeriesKey
type Series = [string, number | null][]
type MetricSeries = { key: SeriesKey; label: string; color: string }

// ────────────────────────────── Metric config ──────────────────────────────

// glossaryKey is the GLOSSARY key for InfoHint; differs from the METRIC_CONFIG entry key for rr
// (glossary: resp_rate) and the device-channel series. Labels are composed from GLOSSARY where the
// label matches the desired display name; the others keep local names because theirs differ.
// hideWhenEmpty marks a series a device may never produce, so isOffered drops its toggle when the
// series is all null. A metric with parts draws each part as its own series on one chart.
const METRIC_CONFIG: Record<
    string,
    {
        label: string
        glossaryKey: string
        hideWhenEmpty?: boolean
        stacked?: boolean
        formatValue?: (value: number) => string
    } & ({ key: SeriesKey; color: string } | { parts: MetricSeries[] })
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
    // OSCAR's Session Times chart (overview.cpp:293-295), as one bedtime and wake time per period.
    session_times: {
        label: GLOSSARY.session_times.label,
        glossaryKey: 'session_times',
        formatValue: formatHoursAfterNoon,
        parts: [
            { key: 'bedtime', label: 'Bedtime', color: '#4338ca' },
            { key: 'wake_time', label: 'Wake', color: '#f59e0b' },
        ],
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
    tv: {
        label: `${GLOSSARY.tidal_volume.label} (mL)`,
        glossaryKey: 'tidal_volume',
        key: 'tv',
        color: '#0891b2',
        hideWhenEmpty: true,
    },
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
    // OSCAR's AHI breakdown chart (overview.cpp:279-285, gAHIChart.h:25-26), stacked from the bottom.
    ahi_breakdown: {
        label: `${GLOSSARY.ahi_breakdown.label} (events/hr)`,
        glossaryKey: 'ahi_breakdown',
        stacked: true,
        parts: [
            { key: 'oai', label: GLOSSARY.oai.label, color: '#be123c' },
            { key: 'cai', label: GLOSSARY.cai.label, color: '#0284c7' },
            { key: 'hi', label: GLOSSARY.hi.label, color: '#ca8a04' },
            { key: 'uai', label: 'UAI', color: '#64748b' },
        ],
    },
    rera: {
        label: 'RERA (events/hr)',
        glossaryKey: 'rera',
        key: 'rera',
        color: '#ea580c',
        hideWhenEmpty: true,
    },
    // OSCAR's TTIA chart (overview.cpp:302-304), as the mean per night with therapy.
    time_in_apnea: {
        label: `${GLOSSARY.time_in_apnea.label} (min)`,
        glossaryKey: 'time_in_apnea',
        key: 'time_in_apnea',
        color: '#9333ea',
    },
    device_flg_95th: {
        label: 'FL 95th',
        glossaryKey: 'fl_device',
        key: 'device_flg_95th',
        color: '#0d9488',
        hideWhenEmpty: true,
    },
    snore_95th: {
        label: 'Snore 95th',
        glossaryKey: 'snore_device',
        key: 'snore_95th',
        color: '#c026d3',
        hideWhenEmpty: true,
    },
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

// FL and snore 95th percentiles come from the periods response, aligned to the trend x-axis by
// period_start; a period missing from the response is null.
const trendSeries = computed<Partial<Record<keyof TrendData | PeriodSeriesKey, Series>> | null>(
    () => {
        if (!trends.value) return null
        const byStart = new Map(periods.value.map((p) => [p.period_start, p]))
        const fromPeriods = (key: PeriodSeriesKey): Series =>
            trendLabels.value.map((label) => [label, byStart.get(label)?.[key] ?? null])
        return {
            ...trends.value,
            device_flg_95th: fromPeriods('device_flg_95th'),
            snore_95th: fromPeriods('snore_95th'),
        }
    },
)

function metricSeries(key: string): MetricSeries[] {
    const cfg = METRIC_CONFIG[key]
    return 'parts' in cfg ? cfg.parts : [{ key: cfg.key, label: cfg.label, color: cfg.color }]
}

function hasData(key: string): boolean {
    if (!METRIC_CONFIG[key] || !trendSeries.value) return false
    return metricSeries(key).some(({ key: seriesKey }) =>
        trendSeries.value?.[seriesKey]?.some((t) => t[1] !== null),
    )
}

function metricDatasets(
    key: string,
): { label: string; values: (number | null)[]; color: string }[] {
    return metricSeries(key).map(({ key: seriesKey, label, color }) => ({
        label,
        values: (trendSeries.value?.[seriesKey] ?? []).map((t) => t[1]),
        color,
    }))
}

const anyVisibleChart = computed(() => selectedMetrics.value.some((key) => hasData(key)))

// An all-null hideWhenEmpty trend means no device in range produces that series, so offer no toggle
// for it. The key stays in selectedMetrics, so the chart returns if a wider range brings data back.
function isOffered(key: string): boolean {
    if (!METRIC_CONFIG[key].hideWhenEmpty || !trendSeries.value) return true
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
