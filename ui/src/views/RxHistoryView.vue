<template>
    <div class="rx-history">
        <h1 class="page-title">RX History</h1>

        <div v-if="loading" class="loading-state">
            <Loader2 class="h-4 w-4 animate-spin" /> Loading RX data...
        </div>

        <ErrorState v-else-if="error" :message="error" :retry="reload" />

        <template v-else>
            <!-- Current Settings -->
            <div v-if="current" class="section-card">
                <h2>Current Settings</h2>
                <div class="current-meta">
                    <span
                        >{{ formatDateFull(current.start_date) }} –
                        {{ formatDateFull(current.end_date) }}</span
                    >
                    <span>{{ current.days_count }} days</span>
                    <span v-if="current.device_name">{{ current.device_name }}</span>
                    <span v-if="current.avg_ahi != null"
                        >Avg AHI: {{ current.avg_ahi.toFixed(1) }}</span
                    >
                    <span v-if="current.avg_hours != null"
                        >Avg {{ current.avg_hours.toFixed(1) }} hrs/night</span
                    >
                </div>
                <div class="settings-pills">
                    <Badge
                        v-for="(value, key) in current.settings"
                        :key="key"
                        variant="secondary"
                        class="setting-pill"
                    >
                        {{ settingLabel(key) }}: {{ formatSettingValue(key, value) }}
                    </Badge>
                </div>
            </div>

            <!-- Comparison Table -->
            <div v-if="history.length > 1" class="section-card">
                <h2>Period Comparison</h2>
                <div v-if="isMobile" class="card-list">
                    <div
                        v-for="row in comparisonRows"
                        :key="row.start_date"
                        class="data-card"
                        :class="{ 'rx-card-best': row.isBest, 'rx-card-worst': row.isWorst }"
                    >
                        <div class="data-card-header">
                            <span
                                >{{ formatDateFull(row.start_date) }} –
                                {{ formatDateFull(row.end_date) }}</span
                            >
                            <Badge
                                v-if="row.isBest"
                                class="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                                >Best</Badge
                            >
                            <Badge v-if="row.isWorst" variant="destructive">Worst</Badge>
                        </div>
                        <div class="data-card-row">
                            <span class="data-card-label">Days</span>
                            <span class="data-card-value">{{ row.days_count }}</span>
                        </div>
                        <div class="data-card-row">
                            <span class="data-card-label">Device</span>
                            <span class="data-card-value">{{ row.device_name ?? '—' }}</span>
                        </div>
                        <div class="data-card-row">
                            <span class="data-card-label">Avg AHI</span>
                            <span class="data-card-value">{{
                                row.avg_ahi?.toFixed(1) ?? '---'
                            }}</span>
                        </div>
                        <div class="data-card-row">
                            <span class="data-card-label">Median AHI</span>
                            <span class="data-card-value">{{
                                row.median_ahi?.toFixed(1) ?? '---'
                            }}</span>
                        </div>
                        <div class="data-card-row">
                            <span class="data-card-label">Avg Hours</span>
                            <span class="data-card-value">{{
                                row.avg_hours?.toFixed(1) ?? '---'
                            }}</span>
                        </div>
                        <div class="data-card-row">
                            <span class="data-card-label">Avg Leak</span>
                            <span class="data-card-value">{{
                                row.avg_leak?.toFixed(1) ?? '---'
                            }}</span>
                        </div>
                        <div class="data-card-label mt-1">Settings</div>
                        <div class="text-sm">{{ summarizeSettings(row.settings) }}</div>
                    </div>
                </div>
                <div v-else class="overflow-x-auto">
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead class="whitespace-nowrap">Period</TableHead>
                                <TableHead class="whitespace-nowrap">Days</TableHead>
                                <TableHead class="whitespace-nowrap">Device</TableHead>
                                <TableHead>Settings</TableHead>
                                <TableHead class="whitespace-nowrap">Avg AHI</TableHead>
                                <TableHead class="whitespace-nowrap">Median AHI</TableHead>
                                <TableHead class="whitespace-nowrap">Avg Hours</TableHead>
                                <TableHead class="whitespace-nowrap">Avg Leak</TableHead>
                                <TableHead></TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            <TableRow
                                v-for="row in comparisonRows"
                                :key="row.start_date"
                                :class="{
                                    'bg-green-500/10': row.isBest,
                                    'bg-destructive/10': row.isWorst,
                                    'even:bg-muted/50': !row.isBest && !row.isWorst,
                                }"
                            >
                                <TableCell class="whitespace-nowrap">
                                    {{ formatDateFull(row.start_date) }} –
                                    {{ formatDateFull(row.end_date) }}
                                </TableCell>
                                <TableCell class="whitespace-nowrap">{{
                                    row.days_count
                                }}</TableCell>
                                <TableCell class="whitespace-nowrap">{{
                                    row.device_name ?? '—'
                                }}</TableCell>
                                <TableCell>{{ summarizeSettings(row.settings) }}</TableCell>
                                <TableCell class="whitespace-nowrap">{{
                                    row.avg_ahi?.toFixed(1) ?? '---'
                                }}</TableCell>
                                <TableCell class="whitespace-nowrap">{{
                                    row.median_ahi?.toFixed(1) ?? '---'
                                }}</TableCell>
                                <TableCell class="whitespace-nowrap">{{
                                    row.avg_hours?.toFixed(1) ?? '---'
                                }}</TableCell>
                                <TableCell class="whitespace-nowrap">{{
                                    row.avg_leak?.toFixed(1) ?? '---'
                                }}</TableCell>
                                <TableCell class="whitespace-nowrap">
                                    <Badge
                                        v-if="row.isBest"
                                        class="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                                        >Best</Badge
                                    >
                                    <Badge v-if="row.isWorst" variant="destructive">Worst</Badge>
                                </TableCell>
                            </TableRow>
                        </TableBody>
                    </Table>
                </div>
            </div>

            <!-- Settings Changes — device changes merged with mask log, most recent first -->
            <div v-if="timelineRows.length || maskError" class="section-card">
                <h2>Settings Changes</h2>
                <div v-if="isMobile && timelineRows.length" class="card-list">
                    <div v-for="row in timelineRows" :key="row.key" class="data-card">
                        <div class="data-card-header">
                            <span>{{ formatDateFull(row.date) }}</span>
                            <Badge v-if="row.source === 'device'" variant="secondary">Device</Badge>
                            <Badge v-else variant="outline">Mask Log</Badge>
                        </div>
                        <template v-if="row.source === 'device'">
                            <div class="data-card-row">
                                <span class="data-card-label">Device</span>
                                <span class="data-card-value">{{ row.change.device_name }}</span>
                            </div>
                            <div class="data-card-row">
                                <span class="data-card-label">Setting</span>
                                <span class="data-card-value">{{
                                    settingLabel(row.change.key)
                                }}</span>
                            </div>
                            <div class="data-card-row">
                                <span class="data-card-label">Change</span>
                                <span class="data-card-value">
                                    <span class="text-muted-foreground">{{
                                        row.change.old_value != null
                                            ? formatSettingValue(
                                                  row.change.key,
                                                  row.change.old_value,
                                              )
                                            : '—'
                                    }}</span>
                                    <span class="mx-1">→</span>
                                    <span>{{
                                        row.change.new_value != null
                                            ? formatSettingValue(
                                                  row.change.key,
                                                  row.change.new_value,
                                              )
                                            : '—'
                                    }}</span>
                                </span>
                            </div>
                        </template>
                        <template v-else>
                            <div class="data-card-row">
                                <span class="data-card-label">Setting</span>
                                <span class="data-card-value">Mask</span>
                            </div>
                            <div class="data-card-row">
                                <span class="data-card-label">Change</span>
                                <span class="data-card-value">{{ maskSummary(row.entry) }}</span>
                            </div>
                        </template>
                    </div>
                </div>
                <div v-else-if="timelineRows.length" class="overflow-x-auto">
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead class="whitespace-nowrap">Date</TableHead>
                                <TableHead class="whitespace-nowrap">Source</TableHead>
                                <TableHead class="whitespace-nowrap">Device</TableHead>
                                <TableHead class="whitespace-nowrap">Setting</TableHead>
                                <TableHead class="whitespace-nowrap">Change</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            <TableRow v-for="row in timelineRows" :key="row.key">
                                <TableCell class="whitespace-nowrap">{{
                                    formatDateFull(row.date)
                                }}</TableCell>
                                <TableCell class="whitespace-nowrap">
                                    <Badge v-if="row.source === 'device'" variant="secondary"
                                        >Device</Badge
                                    >
                                    <Badge v-else variant="outline">Mask Log</Badge>
                                </TableCell>
                                <template v-if="row.source === 'device'">
                                    <TableCell class="whitespace-nowrap">{{
                                        row.change.device_name
                                    }}</TableCell>
                                    <TableCell class="whitespace-nowrap">{{
                                        settingLabel(row.change.key)
                                    }}</TableCell>
                                    <TableCell class="whitespace-nowrap">
                                        <span class="text-muted-foreground">{{
                                            row.change.old_value != null
                                                ? formatSettingValue(
                                                      row.change.key,
                                                      row.change.old_value,
                                                  )
                                                : '—'
                                        }}</span>
                                        <span class="mx-1">→</span>
                                        <span>{{
                                            row.change.new_value != null
                                                ? formatSettingValue(
                                                      row.change.key,
                                                      row.change.new_value,
                                                  )
                                                : '—'
                                        }}</span>
                                    </TableCell>
                                </template>
                                <template v-else>
                                    <TableCell class="whitespace-nowrap">—</TableCell>
                                    <TableCell class="whitespace-nowrap">Mask</TableCell>
                                    <TableCell class="whitespace-nowrap">{{
                                        maskSummary(row.entry)
                                    }}</TableCell>
                                </template>
                            </TableRow>
                        </TableBody>
                    </Table>
                </div>
                <p v-if="maskError" class="text-sm text-destructive mt-2">{{ maskError }}</p>
                <p class="text-sm text-muted-foreground mt-2">
                    Manage your mask equipment on the
                    <RouterLink to="/equipment" class="underline">Equipment page</RouterLink>.
                </p>
            </div>

            <div v-if="!history.length && !timelineRows.length" class="no-data">
                <Info class="h-4 w-4" /> No prescription data available.
            </div>
        </template>
    </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { Loader2, Info } from '@lucide/vue'
import { Badge } from '@/components/ui/badge'
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table'
import { getRxAll } from '@/api/rx'
import { getMaskLog } from '@/api/equipment'
import { useApiLoad } from '@/composables/useApiLoad'
import { useIsMobile } from '@/composables/useIsMobile'
import { formatDateFull } from '@/utils/formatting'
import { settingLabel, formatSettingValue } from '@/utils/deviceSettings'
import { maskEntryName, styleLabel } from '@/utils/maskOptions'
import type { MaskLogEntryResponse, RxPeriodResponse, RxSettingChange } from '@/types'
import ErrorState from '@/components/ErrorState.vue'

const { isMobile } = useIsMobile()

const { data, loading, error, reload } = useApiLoad(() => getRxAll(), 'Failed to load RX data')

const history = computed(() => data.value?.history ?? [])
const current = computed(() => data.value?.current ?? null)

interface ComparisonRow extends RxPeriodResponse {
    isBest: boolean
    isWorst: boolean
}

const comparisonRows = computed<ComparisonRow[]>(() =>
    history.value.map((p, i) => ({
        ...p,
        isBest: data.value?.best_index === i,
        isWorst: data.value?.worst_index === i,
    })),
)

function summarizeSettings(settings: Record<string, string>): string {
    const priorityKeys = [
        'mode',
        'pressure_fixed',
        'pressure_min',
        'pressure_max',
        'ipap',
        'epap',
        'ps',
        'epr_level',
    ]
    const parts: string[] = []
    for (const k of priorityKeys) {
        if (k in settings) {
            parts.push(`${settingLabel(k)}: ${formatSettingValue(k, settings[k])}`)
            if (parts.length === 4) break
        }
    }
    if (!parts.length) {
        return Object.entries(settings)
            .slice(0, 3)
            .map(([k, v]) => `${settingLabel(k)}: ${formatSettingValue(k, v)}`)
            .join(', ')
    }
    return parts.join(', ')
}

// --- Mask log ---

function maskSummary(entry: MaskLogEntryResponse): string {
    const hasName = !!(entry.brand || entry.model)
    const name = maskEntryName(entry)
    const details: string[] = []
    if (hasName && entry.style) details.push(styleLabel(entry.style))
    if (entry.size) details.push(`size ${entry.size}`)
    return details.length ? `${name} (${details.join(', ')})` : name
}

const { data: maskData, error: maskError } = useApiLoad(
    () => getMaskLog(),
    'Failed to load mask log',
)

// Device setting changes merged with mask log entries, most recent first.
type TimelineRow =
    | { source: 'device'; key: string; date: string; change: RxSettingChange }
    | { source: 'mask'; key: string; date: string; entry: MaskLogEntryResponse }

const timelineRows = computed<TimelineRow[]>(() => {
    const deviceRows: TimelineRow[] = (data.value?.changes?.changes ?? []).map((change, i) => ({
        source: 'device',
        key: `device-${change.date}-${change.device_id}-${change.key}-${i}`,
        date: change.date,
        change,
    }))
    const maskRows: TimelineRow[] = (maskData.value ?? [])
        .filter(
            (entry): entry is MaskLogEntryResponse & { start_date: string } => !!entry.start_date,
        )
        .map((entry) => ({
            source: 'mask' as const,
            key: `mask-${entry.id}`,
            date: entry.start_date,
            entry,
        }))
    return [...deviceRows, ...maskRows].sort((a, b) => b.date.localeCompare(a.date))
})
</script>

<style scoped>
.rx-history {
    max-width: 1200px;
}

.no-data {
    padding: 2rem;
    text-align: center;
    color: var(--color-muted-foreground);
}

.current-meta {
    display: flex;
    gap: 1.25rem;
    flex-wrap: wrap;
    font-size: 0.9rem;
    color: var(--color-muted-foreground);
    margin-bottom: 0.75rem;
}

.settings-pills {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
}

.setting-pill {
    font-size: 0.78rem;
}

/* Scoped (higher specificity) so these win over the unlayered .data-card background. */
.rx-card-best {
    background: color-mix(in srgb, var(--color-green-500) 10%, var(--color-card));
}

.rx-card-worst {
    background: color-mix(in srgb, var(--color-destructive) 10%, var(--color-card));
}
</style>
