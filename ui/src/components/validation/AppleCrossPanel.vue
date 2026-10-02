<template>
    <ValidationPanelShell
        validator-type="apple"
        :load-run-id="loadRunId"
        experimental
        experimental-note="Apple Watch sleep signals are a genuinely independent second axis for SNORE's experimental FL/RERA indices. Correlations are noisy validity checks, not calibration."
        :filename-base="fileStem()"
        @update:report="rawReport = $event"
        @download-csv="onDownloadCsv"
    >
        <template #default>
            <div v-if="report" class="space-y-6">
                <div class="grid grid-cols-2 gap-4 lg:grid-cols-4">
                    <StatCard
                        v-for="corr in CORRELATIONS"
                        :key="corr.key"
                        :label="corr.label"
                        :value="report.aggregate[corr.key]?.rho"
                        :decimals="3"
                        :reason="report.aggregate[corr.key]?.reason"
                        :glossary-key="corr.glossaryKey"
                    />
                </div>

                <div class="grid grid-cols-2 gap-4 sm:grid-cols-3">
                    <StatCard
                        label="Total Nights"
                        :value="report.aggregate.total_nights"
                        :decimals="0"
                    />
                    <StatCard
                        label="Nights with Apple BD"
                        :value="report.aggregate.n_with_apple_bd"
                        :decimals="0"
                        glossary-key="apple_breathing_disturbances"
                    />
                    <StatCard
                        label="Skipped (analysis not run)"
                        :value="report.aggregate.n_analysis_not_run"
                        :decimals="0"
                    />
                    <StatCard
                        label="Skipped (device ambiguous)"
                        :value="report.aggregate.n_device_ambiguous"
                        :decimals="0"
                    />
                </div>

                <div class="rounded-md border overflow-x-auto">
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>Night</TableHead>
                                <TableHead class="whitespace-nowrap"
                                    >RERA Index <InfoHint glossary-key="rera_index"
                                /></TableHead>
                                <TableHead class="whitespace-nowrap"
                                    >FL Class ≥4 <InfoHint glossary-key="fl_class_ge4_pct"
                                /></TableHead>
                                <TableHead class="whitespace-nowrap"
                                    >Apple BD <InfoHint glossary-key="apple_breathing_disturbances"
                                /></TableHead>
                                <TableHead class="whitespace-nowrap">Awake (s)</TableHead>
                                <TableHead class="whitespace-nowrap">Sleep Eff %</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            <TableRow v-if="report.nights.length === 0">
                                <TableCell
                                    :colspan="6"
                                    class="py-8 text-center text-muted-foreground"
                                >
                                    No nights in range.
                                </TableCell>
                            </TableRow>
                            <TableRow
                                v-for="n in report.nights"
                                v-else
                                :key="n.night_date"
                                :class="
                                    n.skip_reason ? 'text-muted-foreground' : 'even:bg-muted/50'
                                "
                            >
                                <TableCell>
                                    {{ formatDateMonthDay(n.night_date) }}
                                    <span
                                        v-if="n.skip_reason"
                                        class="ml-1 text-xs"
                                        :title="nullReasonLabel(n.skip_reason) ?? undefined"
                                        >({{ n.skip_reason }})</span
                                    >
                                </TableCell>
                                <TableCell>{{ num(n.rera_index, 2) }}</TableCell>
                                <TableCell>{{ num(n.fl_class_ge4_pct, 1) }}</TableCell>
                                <TableCell>{{ num(n.apple_breathing_disturbances, 2) }}</TableCell>
                                <TableCell>{{ num(n.awake_seconds, 0) }}</TableCell>
                                <TableCell>{{ num(n.sleep_efficiency_pct, 1) }}</TableCell>
                            </TableRow>
                        </TableBody>
                    </Table>
                </div>
            </div>
        </template>
    </ValidationPanelShell>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import StatCard from '@/components/StatCard.vue'
import InfoHint from '@/components/InfoHint.vue'
import ValidationPanelShell from '@/components/validation/ValidationPanelShell.vue'
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table'
import { formatDateMonthDay, nullReasonLabel } from '@/utils/formatting'
import { downloadCsv } from '@/utils/download'
import type { AppleCrossValidationReport } from '@/types'

defineProps<{ loadRunId?: number | null }>()

type CorrKey =
    'rera_vs_apple_bd' | 'fl_vs_apple_bd' | 'rera_vs_awake_seconds' | 'fl_vs_sleep_efficiency'
const CORRELATIONS: { key: CorrKey; label: string; glossaryKey: string }[] = [
    {
        key: 'rera_vs_apple_bd',
        label: 'RERA vs Apple BD',
        glossaryKey: 'apple_breathing_disturbances',
    },
    { key: 'fl_vs_apple_bd', label: 'FL vs Apple BD', glossaryKey: 'apple_breathing_disturbances' },
    { key: 'rera_vs_awake_seconds', label: 'RERA vs Awake', glossaryKey: 'spearman_r' },
    { key: 'fl_vs_sleep_efficiency', label: 'FL vs Sleep Eff', glossaryKey: 'spearman_r' },
]

const rawReport = ref<Record<string, unknown> | null>(null)
const report = computed(() => rawReport.value as AppleCrossValidationReport | null)

function num(v: number | null | undefined, decimals: number): string {
    return v != null ? v.toFixed(decimals) : '—'
}

function fileStem(): string {
    const r = report.value
    return r ? `apple-cross-${r.date_range_start}-${r.date_range_end}` : 'apple-cross'
}

function onDownloadCsv(): void {
    const r = report.value
    if (!r) return
    const headers = [
        'night_date',
        'rera_index',
        'fl_class_ge4_pct',
        'apple_breathing_disturbances',
        'awake_seconds',
        'sleep_efficiency_pct',
        'skip_reason',
    ]
    const cell = (v: number | null | undefined): string => (v != null ? v.toFixed(4) : '')
    const rows = r.nights.map((n) => [
        n.night_date,
        cell(n.rera_index),
        cell(n.fl_class_ge4_pct),
        cell(n.apple_breathing_disturbances),
        cell(n.awake_seconds),
        cell(n.sleep_efficiency_pct),
        n.skip_reason ?? '',
    ])
    downloadCsv(headers, rows, `${fileStem()}.csv`)
}
</script>
