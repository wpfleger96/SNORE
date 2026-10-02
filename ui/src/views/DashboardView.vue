<template>
    <div class="dashboard">
        <h1 class="page-title">Dashboard</h1>

        <template v-if="loading">
            <div class="summary-row">
                <Skeleton v-for="i in 4" :key="'a' + i" class="h-[88px] rounded-lg" />
            </div>
            <div class="summary-row">
                <Skeleton v-for="i in 4" :key="'b' + i" class="h-[88px] rounded-lg" />
            </div>
            <div class="section-card">
                <Skeleton class="h-5 w-40 mb-4" />
                <Skeleton class="h-[280px] w-full rounded-lg" />
            </div>
            <div class="section-card">
                <Skeleton class="h-5 w-36 mb-4" />
                <Skeleton class="h-[120px] w-full rounded-lg" />
            </div>
        </template>

        <!-- Summary Cards -->
        <div v-if="summary && !loading" class="summary-row">
            <StatCard
                label="Days with Data"
                :value="summary.days_with_data"
                :decimals="0"
                glossary-key="days_with_data"
            />
            <div class="stat-card-ahi">
                <StatCard
                    label="Avg AHI"
                    :value="summary.avg_ahi"
                    :decimals="1"
                    glossary-key="ahi"
                />
                <Badge
                    v-if="summary.effectiveness !== 'unknown'"
                    v-bind="effectivenessBadgeAttrs(summary.effectiveness)"
                    class="effectiveness-badge"
                >
                    {{ summary.effectiveness }}<InfoHint glossary-key="effectiveness" />
                </Badge>
                <span
                    v-if="summary.ahi_trend_direction"
                    class="trend-badge"
                    :class="'trend-' + summary.ahi_trend_direction"
                >
                    {{ summary.ahi_trend_direction }}<InfoHint glossary-key="ahi_trend" />
                </span>
            </div>
            <StatCard
                label="Avg Hours"
                :value="summary.avg_hours"
                unit="hrs"
                :decimals="1"
                glossary-key="usage"
            />
            <StatCard
                label="Avg Leak"
                :value="summary.avg_leak"
                unit="L/min"
                :decimals="1"
                glossary-key="leak"
            />
        </div>
        <div v-if="summary && !loading" class="summary-row">
            <StatCard
                label="Avg SpO₂"
                :value="summary.avg_spo2"
                unit="%"
                :decimals="1"
                glossary-key="spo2"
            />
            <StatCard
                label="Avg Pulse"
                :value="summary.avg_pulse"
                unit="bpm"
                :decimals="0"
                glossary-key="pulse"
            />
            <StatCard
                label="Avg Pressure"
                :value="summary.avg_pressure"
                unit="cmH₂O"
                :decimals="1"
                glossary-key="pressure"
            />
            <StatCard
                label="Avg Resp Rate"
                :value="summary.avg_respiratory_rate"
                unit="br/min"
                :decimals="1"
                glossary-key="resp_rate"
            />
        </div>
        <div v-else-if="!loading" class="no-data">No therapy data available.</div>
        <!-- Apple Health sleep summary row -->
        <div
            v-if="summary && !loading && (avgTotalSleepHours != null || avgSleepEfficiency != null)"
            class="summary-row"
        >
            <StatCard
                label="Avg Sleep"
                :value="avgTotalSleepHours"
                unit="hrs"
                :decimals="1"
                glossary-key="total_sleep"
            />
            <StatCard
                label="Avg Sleep Efficiency"
                :value="avgSleepEfficiency"
                unit="%"
                :decimals="1"
                glossary-key="sleep_efficiency"
            />
        </div>

        <div v-if="summary?.event_counts?.length" class="section-card">
            <h2>
                Event Breakdown <InfoHint label="Event Types"><EventTypeLegend /></InfoHint>
            </h2>
            <div class="event-breakdown">
                <div v-for="ec in summary.event_counts" :key="ec.event_type" class="event-item">
                    <Badge
                        :style="{
                            backgroundColor: eventColor(ec.event_type),
                            color: eventTextColor(ec.event_type),
                        }"
                    >
                        {{ ec.event_type }}
                    </Badge>
                    <span class="event-count">{{ ec.count }}</span>
                    <span v-if="ec.percentage != null" class="text-muted-foreground text-sm"
                        >({{ ec.percentage.toFixed(1) }}%)</span
                    >
                </div>
            </div>
        </div>

        <!-- AHI Trend Chart -->
        <div v-if="trendLabels.length" class="section-card">
            <h2>AHI Trend (Weekly)</h2>
            <TrendChart :labels="trendLabels" :datasets="trendDatasets" />
        </div>

        <!-- Calendar Heatmap -->
        <div v-if="days.length" class="section-card">
            <h2>
                Usage Calendar
                <InfoHint label="AHI Color Scale">
                    <ul class="space-y-1 text-xs">
                        <li
                            v-for="entry in AHI_COLOR_SCALE"
                            :key="entry.label"
                            class="flex items-center gap-1.5"
                        >
                            <span
                                class="h-3 w-3 shrink-0 rounded-sm"
                                :style="{ background: entry.color }"
                            />
                            <span>{{ entry.label }}</span>
                        </li>
                        <li class="flex items-center gap-1.5">
                            <span class="h-3 w-3 shrink-0 rounded-sm bg-muted" />
                            <span class="text-muted-foreground">No data</span>
                        </li>
                    </ul>
                    <p class="text-xs text-muted-foreground mt-2">
                        Note: this display scale is stricter than the common clinical convention
                        (&lt;5 normal, 5–15 mild, 15–30 moderate, &gt;30 severe).
                    </p>
                </InfoHint>
            </h2>
            <CalendarHeatmap :days="days" :months-back="isMobile ? 3 : 6" @day-click="onDayClick" />
        </div>

        <!-- Recent Sessions -->
        <div v-if="recentSessions.length" class="section-card">
            <h2>Recent Sessions</h2>
            <div v-if="isMobile" class="card-list">
                <RouterLink
                    v-for="session in recentSessions"
                    :key="session.id"
                    class="data-card"
                    :to="{ name: 'session-detail', params: { id: session.id } }"
                >
                    <div class="data-card-header">{{ formatDateFull(session.therapy_day) }}</div>
                    <div class="data-card-row">
                        <span class="data-card-label">Duration</span>
                        <span class="data-card-value"
                            >{{ session.duration_hours.toFixed(1) }}h</span
                        >
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label">AHI</span>
                        <span class="data-card-value">{{ session.ahi?.toFixed(1) ?? '---' }}</span>
                    </div>
                </RouterLink>
            </div>
            <Table v-else>
                <TableHeader>
                    <TableRow>
                        <TableHead>Date</TableHead>
                        <TableHead class="w-[90px]">Duration</TableHead>
                        <TableHead class="w-[80px] whitespace-nowrap"
                            >AHI <InfoHint glossary-key="ahi"
                        /></TableHead>
                    </TableRow>
                </TableHeader>
                <TableBody>
                    <TableRow
                        v-for="session in recentSessions"
                        :key="session.id"
                        class="cursor-pointer even:bg-muted/50 hover:bg-muted/50"
                        @click="navigateToSession(session)"
                    >
                        <TableCell>{{ formatDateFull(session.therapy_day) }}</TableCell>
                        <TableCell>{{ session.duration_hours.toFixed(1) }}h</TableCell>
                        <TableCell>{{ session.ahi?.toFixed(1) ?? '---' }}</TableCell>
                    </TableRow>
                </TableBody>
            </Table>
        </div>
    </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { RouterLink, useRouter } from 'vue-router'
import { Badge } from '@/components/ui/badge'
import { Skeleton } from '@/components/ui/skeleton'
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table'
import StatCard from '@/components/StatCard.vue'
import InfoHint from '@/components/InfoHint.vue'
import EventTypeLegend from '@/components/EventTypeLegend.vue'
import TrendChart from '@/components/TrendChart.vue'
import CalendarHeatmap from '@/components/CalendarHeatmap.vue'
import { getSummary, getTrends } from '@/api/stats'
import { getDays } from '@/api/days'
import { getSessions } from '@/api/sessions'
import { getHealthNights } from '@/api/health'
import { useApiLoad } from '@/composables/useApiLoad'
import { useIsMobile } from '@/composables/useIsMobile'
import { formatDateFull } from '@/utils/formatting'
import { AHI_COLOR_SCALE } from '@/utils/ahiScale'
import { EVENT_COLORS } from '@/types'
import type { HealthNightSummaryRead, SessionListItem } from '@/types'

function thirtyDaysAgo(): string {
    const d = new Date()
    d.setDate(d.getDate() - 30)
    return d.toISOString().split('T')[0]
}

const router = useRouter()
const { isMobile } = useIsMobile()

// Dashboard gracefully shows empty sections on error, so `error` is unused.
const { data, loading } = useApiLoad(async () => {
    const [summaryRes, trendsRes, daysRes, sessionsRes, healthRes] = await Promise.allSettled([
        getSummary(),
        getTrends('week'),
        getDays({ limit: 365 }),
        getSessions({ limit: 5, sort_by: 'date-desc' }),
        getHealthNights({ limit: 30, from_date: thirtyDaysAgo() }),
    ])
    return {
        summary: summaryRes.status === 'fulfilled' ? summaryRes.value : null,
        trends: trendsRes.status === 'fulfilled' ? trendsRes.value : null,
        days: daysRes.status === 'fulfilled' ? daysRes.value.items : null,
        recentSessions: sessionsRes.status === 'fulfilled' ? sessionsRes.value.items : null,
        healthNights: healthRes.status === 'fulfilled' ? healthRes.value.items : null,
    }
})

const summary = computed(() => data.value?.summary ?? null)
const trends = computed(() => data.value?.trends ?? null)
const days = computed(() => data.value?.days ?? [])
const recentSessions = computed(() => data.value?.recentSessions ?? [])
const healthNights = computed(() => data.value?.healthNights ?? null)

const avgTotalSleepHours = computed(() => {
    const nights = healthNights.value as HealthNightSummaryRead[] | null
    if (!nights?.length) return null
    const valid = nights.filter((n) => n.total_sleep_seconds != null)
    if (!valid.length) return null
    return valid.reduce((sum, n) => sum + n.total_sleep_seconds! / 3600, 0) / valid.length
})

const avgSleepEfficiency = computed(() => {
    const nights = healthNights.value as HealthNightSummaryRead[] | null
    if (!nights?.length) return null
    const valid = nights.filter((n) => n.sleep_efficiency_pct != null)
    if (!valid.length) return null
    return valid.reduce((sum, n) => sum + n.sleep_efficiency_pct!, 0) / valid.length
})

const trendLabels = computed(() => trends.value?.ahi.map((t) => t[0]) ?? [])
const trendDatasets = computed(() => {
    if (!trends.value) return []
    const datasets: { label: string; values: (number | null)[]; color: string }[] = [
        { label: 'AHI', values: trends.value.ahi.map((t) => t[1]), color: '#2563eb' },
        { label: 'Usage (hrs)', values: trends.value.usage.map((t) => t[1]), color: '#16a34a' },
        { label: 'SpO₂ (%)', values: trends.value.spo2.map((t) => t[1]), color: '#f59e0b' },
        { label: 'Leak (L/min)', values: trends.value.leak.map((t) => t[1]), color: '#ef4444' },
    ]
    const sleepEff = trends.value.sleep_efficiency
    if (sleepEff && sleepEff.some((t) => t[1] != null)) {
        datasets.push({
            label: 'Sleep Efficiency (%)',
            values: sleepEff.map((t) => t[1]),
            color: '#8b5cf6',
        })
    }
    return datasets
})

function effectivenessBadgeAttrs(e: string): Record<string, string> {
    if (e === 'excellent' || e === 'good')
        return { class: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200' }
    if (e === 'fair')
        return { class: 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200' }
    if (e === 'poor') return { variant: 'destructive' }
    return { variant: 'secondary' }
}

function navigateToSession(session: SessionListItem): void {
    void router.push({ name: 'session-detail', params: { id: session.id } })
}

function onDayClick(date: string): void {
    void router.push({ name: 'day-detail', params: { date } })
}

function eventColor(type: string): string {
    return EVENT_COLORS[type] ?? 'rgba(156, 163, 175, 0.25)'
}

function eventTextColor(type: string): string {
    return EVENT_COLORS[type] ? 'inherit' : 'var(--color-muted-foreground)'
}
</script>

<style scoped>
.dashboard {
    max-width: 1200px;
}

.summary-row {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1.5rem;
}

.stat-card-ahi {
    position: relative;
}

.effectiveness-badge {
    position: absolute;
    top: 0.5rem;
    right: 0.5rem;
}

.trend-badge {
    position: absolute;
    bottom: 0.5rem;
    right: 0.5rem;
    font-size: 0.7rem;
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
    color: var(--muted-foreground);
}

.no-data {
    padding: 2rem;
    text-align: center;
    color: var(--color-muted-foreground);
}

.event-breakdown {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem;
}

.event-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.event-count {
    font-weight: 600;
}
</style>
