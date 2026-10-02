<template>
    <ValidationPanelShell
        validator-type="breaths"
        :load-run-id="loadRunId"
        experimental
        experimental-note="SNORE's own breath segmentation is compared against the device's breath-level channels. Correlations show whether it tracks the device night to night, not absolute accuracy."
        :filename-base="fileStem()"
        @update:report="rawReport = $event"
        @download-csv="onDownloadCsv"
    >
        <template #default>
            <div v-if="report" class="space-y-6">
                <div class="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
                    <StatCard
                        v-for="ch in CHANNELS"
                        :key="ch.key"
                        :label="`${ch.label} Spearman`"
                        :value="report.aggregate[ch.key].mean_spearman_r"
                        :decimals="3"
                        glossary-key="spearman_r"
                        field="mean_spearman_r"
                    />
                    <StatCard
                        label="Sessions Compared"
                        :value="report.aggregate.sessions_compared"
                        :decimals="0"
                    />
                </div>

                <div class="rounded-md border overflow-x-auto">
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>Date</TableHead>
                                <TableHead class="whitespace-nowrap"
                                    >Breaths
                                    <ProvenanceMark :provenance="provenanceFor('n_breaths')"
                                /></TableHead>
                                <TableHead
                                    v-for="ch in CHANNELS"
                                    :key="ch.key"
                                    class="whitespace-nowrap"
                                >
                                    {{ ch.label }}
                                    <ProvenanceMark :provenance="provenanceFor('spearman_r')" />
                                    <InfoHint glossary-key="spearman_r" />
                                </TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            <TableRow v-if="report.sessions.length === 0">
                                <TableCell
                                    :colspan="CHANNELS.length + 2"
                                    class="py-8 text-center text-muted-foreground"
                                >
                                    No sessions in range.
                                </TableCell>
                            </TableRow>
                            <TableRow
                                v-for="s in report.sessions"
                                v-else
                                :key="s.session_id"
                                class="even:bg-muted/50"
                            >
                                <TableCell>
                                    <SessionDateCell
                                        :session-id="s.session_id"
                                        :date="s.date"
                                        :skipped-reason="s.skipped_reason"
                                    />
                                </TableCell>
                                <TableCell>{{ s.n_breaths }}</TableCell>
                                <TableCell v-for="ch in CHANNELS" :key="ch.key">
                                    {{ channelSpearman(s, ch.key) }}
                                </TableCell>
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
import ProvenanceMark from '@/components/ProvenanceMark.vue'
import ValidationPanelShell from '@/components/validation/ValidationPanelShell.vue'
import SessionDateCell from '@/components/validation/SessionDateCell.vue'
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table'
import { downloadCsv } from '@/utils/download'
import { provenanceFor } from '@/utils/provenance'
import type { BreathTrendsValidationReport, BreathTrendsSessionValidation } from '@/types'

defineProps<{ loadRunId?: number | null }>()

type ChannelKey = 'rr' | 'tv' | 'ti' | 'ie_ratio'
const CHANNELS: { key: ChannelKey; label: string }[] = [
    { key: 'rr', label: 'RR' },
    { key: 'tv', label: 'TV' },
    { key: 'ti', label: 'Ti' },
    { key: 'ie_ratio', label: 'I:E' },
]

const rawReport = ref<Record<string, unknown> | null>(null)
const report = computed(() => rawReport.value as BreathTrendsValidationReport | null)

function channelSpearman(s: BreathTrendsSessionValidation, key: ChannelKey): string {
    const ch = s.channels?.[key]
    if (!ch || ch.spearman_r == null) return '—'
    return ch.spearman_r.toFixed(3)
}

function fileStem(): string {
    const r = report.value
    return r ? `breaths-validation-${r.date_range_start}-${r.date_range_end}` : 'breaths-validation'
}

function onDownloadCsv(): void {
    const r = report.value
    if (!r) return
    const headers = [
        'session_id',
        'date',
        'duration_hours',
        'skipped_reason',
        'n_breaths',
        'channel',
        'n_pairs',
        'spearman_r',
        'median_abs_error',
        'mean_bias',
        'channel_skipped_reason',
    ]
    const cell = (v: number | null | undefined): string => (v != null ? v.toFixed(4) : '')
    const rows: unknown[][] = []
    for (const s of r.sessions) {
        for (const { key } of CHANNELS) {
            const ch = s.channels?.[key]
            rows.push([
                s.session_id,
                s.date,
                s.duration_hours.toFixed(1),
                s.skipped_reason ?? '',
                s.n_breaths,
                key,
                ch?.n_pairs ?? '',
                cell(ch?.spearman_r),
                cell(ch?.median_abs_error),
                cell(ch?.mean_bias),
                ch?.skipped_reason ?? '',
            ])
        }
    }
    downloadCsv(headers, rows, `${fileStem()}.csv`)
}
</script>
