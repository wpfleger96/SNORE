<template>
    <div class="relative">
        <div
            v-if="loading"
            class="absolute inset-0 z-10 flex items-center justify-center bg-background/60"
        >
            <Loader2 class="h-6 w-6 animate-spin text-muted-foreground" />
        </div>
        <template v-if="isMobile">
            <div v-if="periods.length" class="card-list">
                <div v-for="row in periods" :key="row.period_start" class="data-card">
                    <div class="data-card-header">
                        <template v-if="row.period_start === row.period_end">
                            {{ formatDateMonthDay(row.period_start) }}
                        </template>
                        <template v-else>
                            {{ formatDateMonthDay(row.period_start) }} –
                            {{ formatDateMonthDay(row.period_end) }}
                        </template>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label">Days Used</span>
                        <span class="data-card-value">{{ row.days_used }}</span>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label"
                            >Avg Hours <InfoHint glossary-key="usage"
                        /></span>
                        <span class="data-card-value">{{
                            row.avg_hours_per_day?.toFixed(1) ?? '---'
                        }}</span>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label">Avg AHI <InfoHint glossary-key="ahi" /></span>
                        <span class="data-card-value" :class="ahiClass(row.avg_ahi)">
                            {{ row.avg_ahi?.toFixed(1) ?? '---' }}
                        </span>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label">Median AHI</span>
                        <span class="data-card-value">{{
                            row.median_ahi?.toFixed(1) ?? '---'
                        }}</span>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label"
                            >Avg Pressure <InfoHint glossary-key="pressure"
                        /></span>
                        <span class="data-card-value">{{
                            row.avg_pressure?.toFixed(1) ?? '---'
                        }}</span>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label"
                            >Avg Leak <InfoHint glossary-key="leak"
                        /></span>
                        <span class="data-card-value">{{ row.avg_leak?.toFixed(1) ?? '---' }}</span>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label"
                            >Avg SpO₂ <InfoHint glossary-key="spo2"
                        /></span>
                        <span class="data-card-value">{{ row.avg_spo2?.toFixed(1) ?? '---' }}</span>
                    </div>
                    <div v-if="showSleepColumns" class="data-card-row">
                        <span class="data-card-label"
                            >Avg Sleep <InfoHint glossary-key="total_sleep"
                        /></span>
                        <span class="data-card-value">{{
                            row.avg_total_sleep_hours?.toFixed(1) ?? '---'
                        }}</span>
                    </div>
                    <div v-if="showSleepColumns" class="data-card-row">
                        <span class="data-card-label"
                            >Avg Eff <InfoHint glossary-key="sleep_efficiency"
                        /></span>
                        <span class="data-card-value">{{
                            row.avg_sleep_efficiency_pct?.toFixed(1) ?? '---'
                        }}</span>
                    </div>
                </div>
            </div>
            <div
                v-else
                class="flex h-24 items-center justify-center text-center text-muted-foreground"
            >
                {{ emptyMessage ?? 'No period data available.' }}
            </div>
        </template>
        <Table v-else>
            <TableHeader>
                <TableRow>
                    <TableHead>Period</TableHead>
                    <TableHead class="w-[90px]">Days Used</TableHead>
                    <TableHead class="w-[90px] whitespace-nowrap"
                        >Avg Hours <InfoHint glossary-key="usage"
                    /></TableHead>
                    <TableHead class="w-[90px] whitespace-nowrap"
                        >Avg AHI <InfoHint glossary-key="ahi"
                    /></TableHead>
                    <TableHead class="w-[100px]">Median AHI</TableHead>
                    <TableHead class="w-[110px] whitespace-nowrap"
                        >Avg Pressure <InfoHint glossary-key="pressure"
                    /></TableHead>
                    <TableHead class="w-[90px] whitespace-nowrap"
                        >Avg Leak <InfoHint glossary-key="leak"
                    /></TableHead>
                    <TableHead class="w-[90px] whitespace-nowrap"
                        >Avg SpO₂ <InfoHint glossary-key="spo2"
                    /></TableHead>
                    <TableHead v-if="showSleepColumns" class="w-[100px] whitespace-nowrap"
                        >Avg Sleep <InfoHint glossary-key="total_sleep"
                    /></TableHead>
                    <TableHead v-if="showSleepColumns" class="w-[90px] whitespace-nowrap"
                        >Avg Eff <InfoHint glossary-key="sleep_efficiency"
                    /></TableHead>
                </TableRow>
            </TableHeader>
            <TableBody>
                <template v-if="periods.length">
                    <TableRow
                        v-for="row in periods"
                        :key="row.period_start"
                        class="even:bg-muted/50"
                    >
                        <TableCell>
                            <template v-if="row.period_start === row.period_end">
                                {{ formatDateMonthDay(row.period_start) }}
                            </template>
                            <template v-else>
                                {{ formatDateMonthDay(row.period_start) }} –
                                {{ formatDateMonthDay(row.period_end) }}
                            </template>
                        </TableCell>
                        <TableCell>{{ row.days_used }}</TableCell>
                        <TableCell>{{ row.avg_hours_per_day?.toFixed(1) ?? '---' }}</TableCell>
                        <TableCell>
                            <span :class="ahiClass(row.avg_ahi)">
                                {{ row.avg_ahi?.toFixed(1) ?? '---' }}
                            </span>
                        </TableCell>
                        <TableCell>{{ row.median_ahi?.toFixed(1) ?? '---' }}</TableCell>
                        <TableCell>{{ row.avg_pressure?.toFixed(1) ?? '---' }}</TableCell>
                        <TableCell>{{ row.avg_leak?.toFixed(1) ?? '---' }}</TableCell>
                        <TableCell>{{ row.avg_spo2?.toFixed(1) ?? '---' }}</TableCell>
                        <TableCell v-if="showSleepColumns">{{
                            row.avg_total_sleep_hours?.toFixed(1) ?? '---'
                        }}</TableCell>
                        <TableCell v-if="showSleepColumns">{{
                            row.avg_sleep_efficiency_pct?.toFixed(1) ?? '---'
                        }}</TableCell>
                    </TableRow>
                </template>
                <TableRow v-else>
                    <TableCell
                        :colspan="showSleepColumns ? 10 : 8"
                        class="h-24 text-center text-muted-foreground"
                    >
                        {{ emptyMessage ?? 'No period data available.' }}
                    </TableCell>
                </TableRow>
            </TableBody>
        </Table>
    </div>
</template>

<script setup lang="ts">
import { Loader2 } from '@lucide/vue'
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table'
import InfoHint from '@/components/InfoHint.vue'
import { useIsMobile } from '@/composables/useIsMobile'
import type { PeriodStatistics } from '@/types'
import { ahiClass, formatDateMonthDay } from '@/utils/formatting'

defineProps<{
    periods: PeriodStatistics[]
    loading: boolean
    emptyMessage?: string
    showSleepColumns?: boolean
}>()

const { isMobile } = useIsMobile()
</script>
