<template>
    <ValidationPanelShell
        validator-type="rera"
        :load-run-id="loadRunId"
        experimental
        experimental-note="SNORE's RERA proxy is an experimental trend instrument, not a validated event count. It is scored against the device-flagged RE events, which the device reports extremely conservatively."
        :filename-base="fileStem()"
        @update:report="rawReport = $event"
        @download-csv="onDownloadCsv"
    >
        <template #default>
            <div v-if="report" class="space-y-6">
                <!-- Framing: precision near the chance floor is expected, not failure. -->
                <div
                    class="rounded-md border border-border bg-muted/40 px-4 py-3 text-sm text-muted-foreground"
                >
                    <p class="mb-1 font-medium text-foreground">How to read these scores</p>
                    <p>
                        The device flags RERAs very conservatively, so the FL-run proxy fires far
                        more often than there are device RE events to match. Low sensitivity and
                        near-zero precision are the
                        <span class="font-medium">expected</span> result, not a detector failure.
                        Compare proxy precision to the
                        <InfoHint glossary-key="chance_floor" /> chance precision floor below:
                        precision at or below the floor is
                        <span class="font-medium">indistinguishable from chance</span> given how
                        often the proxy fires.
                    </p>
                </div>

                <div class="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
                    <StatCard
                        label="Proxy Sensitivity"
                        :value="report.aggregate.mean_proxy_sensitivity"
                        field="ReraAggregateMetrics.mean_proxy_sensitivity"
                        :display="formatPercent(report.aggregate.mean_proxy_sensitivity)"
                        glossary-key="sensitivity"
                    />
                    <StatCard
                        label="Proxy Precision"
                        :value="report.aggregate.mean_proxy_precision"
                        field="ReraAggregateMetrics.mean_proxy_precision"
                        :display="formatPercent(report.aggregate.mean_proxy_precision, 2)"
                        glossary-key="precision"
                    />
                    <StatCard
                        label="Chance Precision Floor"
                        :value="report.aggregate.chance_precision_floor"
                        field="ReraAggregateMetrics.chance_precision_floor"
                        :display="formatPercent(report.aggregate.chance_precision_floor, 2)"
                        glossary-key="chance_floor"
                        :reason="floorReason"
                    />
                    <StatCard
                        label="Amplitude Sensitivity"
                        :value="report.aggregate.mean_amplitude_sensitivity"
                        field="ReraAggregateMetrics.mean_amplitude_sensitivity"
                        :display="formatPercent(report.aggregate.mean_amplitude_sensitivity)"
                        glossary-key="sensitivity"
                    />
                    <StatCard
                        label="Amplitude Precision"
                        :value="report.aggregate.mean_amplitude_precision"
                        field="ReraAggregateMetrics.mean_amplitude_precision"
                        :display="formatPercent(report.aggregate.mean_amplitude_precision, 2)"
                        glossary-key="precision"
                    />
                    <StatCard
                        label="Device RE Density"
                        :value="report.aggregate.machine_re_density"
                        field="ReraAggregateMetrics.machine_re_density"
                        :decimals="2"
                        unit="/h"
                    />
                    <StatCard
                        label="Proxy Density"
                        :value="report.aggregate.proxy_density"
                        field="ReraAggregateMetrics.proxy_density"
                        :decimals="2"
                        unit="/h"
                        glossary-key="rera_proxy"
                    />
                </div>

                <p class="text-sm font-medium">
                    <span :class="atFloor ? 'text-muted-foreground' : 'text-foreground'">
                        {{ floorVerdict }}
                    </span>
                </p>

                <div class="grid grid-cols-2 gap-4 sm:grid-cols-4">
                    <StatCard
                        label="Total Device RE"
                        :value="report.aggregate.total_machine_re"
                        :decimals="0"
                    />
                    <StatCard
                        label="Total Proxy RERAs"
                        :value="report.aggregate.total_proxy_reras"
                        field="ReraAggregateMetrics.total_proxy_reras"
                        :decimals="0"
                    />
                    <StatCard
                        label="Scored Sessions"
                        :value="report.aggregate.sessions_with_machine_re"
                        :decimals="0"
                    />
                    <StatCard
                        label="Skipped (No Device RE)"
                        :value="report.aggregate.sessions_skipped_no_machine_re"
                        :decimals="0"
                    />
                </div>

                <div class="rounded-md border overflow-x-auto">
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>Date</TableHead>
                                <TableHead class="whitespace-nowrap">Device RE</TableHead>
                                <TableHead class="whitespace-nowrap"
                                    >Amplitude RERAs
                                    <ProvenanceMark
                                        :provenance="
                                            provenanceFor(
                                                'ReraSessionValidation.amplitude_rera_count',
                                            )
                                        "
                                /></TableHead>
                                <TableHead class="whitespace-nowrap"
                                    >Proxy RERAs
                                    <ProvenanceMark
                                        :provenance="
                                            provenanceFor('ReraSessionValidation.proxy_rera_count')
                                        "
                                /></TableHead>
                                <TableHead class="whitespace-nowrap"
                                    >Proxy Sens
                                    <ProvenanceMark
                                        :provenance="
                                            provenanceFor('ReraSessionValidation.proxy_sensitivity')
                                        " />
                                    <InfoHint glossary-key="sensitivity"
                                /></TableHead>
                                <TableHead class="whitespace-nowrap"
                                    >Proxy Prec
                                    <ProvenanceMark
                                        :provenance="
                                            provenanceFor('ReraSessionValidation.proxy_precision')
                                        " />
                                    <InfoHint glossary-key="precision"
                                /></TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            <TableRow v-if="report.sessions.length === 0">
                                <TableCell
                                    :colspan="6"
                                    class="py-8 text-center text-muted-foreground"
                                >
                                    No sessions in range.
                                </TableCell>
                            </TableRow>
                            <TableRow
                                v-for="s in report.sessions"
                                v-else
                                :key="s.session_id"
                                :class="
                                    s.skipped_reason ? 'text-muted-foreground' : 'even:bg-muted/50'
                                "
                            >
                                <TableCell>
                                    <SessionDateCell
                                        :session-id="s.session_id"
                                        :date="s.date"
                                        :skipped-reason="s.skipped_reason"
                                        show-reason
                                    />
                                </TableCell>
                                <TableCell>{{ s.machine_re_count }}</TableCell>
                                <TableCell>{{ s.amplitude_rera_count }}</TableCell>
                                <TableCell>{{ s.proxy_rera_count }}</TableCell>
                                <TableCell>{{
                                    formatPercent(s.proxy_sensitivity) ?? '—'
                                }}</TableCell>
                                <TableCell>{{
                                    formatPercent(s.proxy_precision, 2) ?? '—'
                                }}</TableCell>
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
import { formatPercent } from '@/utils/formatting'
import { downloadCsv } from '@/utils/download'
import { provenanceFor } from '@/utils/provenance'
import type { ReraValidationReport } from '@/types'

defineProps<{ loadRunId?: number | null }>()

const rawReport = ref<Record<string, unknown> | null>(null)
const report = computed(() => rawReport.value as ReraValidationReport | null)

const atFloor = computed<boolean>(() => {
    const agg = report.value?.aggregate
    if (!agg || agg.mean_proxy_precision == null || agg.chance_precision_floor == null) {
        return false
    }
    return agg.mean_proxy_precision <= agg.chance_precision_floor
})

const floorReason = computed<string | null>(() =>
    report.value?.aggregate.chance_precision_floor == null ? 'no_data_in_range' : null,
)

const floorVerdict = computed<string>(() => {
    const agg = report.value?.aggregate
    if (!agg || agg.mean_proxy_precision == null || agg.chance_precision_floor == null) {
        return 'Not enough scored therapy hours to compare proxy precision against the chance floor.'
    }
    return atFloor.value
        ? 'Proxy precision is at or below the chance floor — indistinguishable from chance given the proxy firing density. This is the expected result.'
        : 'Proxy precision exceeds the chance floor — the proxy carries signal above random firing.'
})

function fileStem(): string {
    const r = report.value
    return r ? `rera-validation-${r.date_range_start}-${r.date_range_end}` : 'rera-validation'
}

function onDownloadCsv(): void {
    const r = report.value
    if (!r) return
    const headers = [
        'session_id',
        'date',
        'duration_hours',
        'skipped_reason',
        'machine_re_count',
        'amplitude_rera_count',
        'proxy_rera_count',
        'proxy_sensitivity',
        'proxy_precision',
        'proxy_f1',
    ]
    const cell = (v: number | null | undefined): string => (v != null ? v.toFixed(4) : '')
    const rows = r.sessions.map((s) => [
        s.session_id,
        s.date,
        s.duration_hours.toFixed(1),
        s.skipped_reason ?? '',
        s.machine_re_count,
        s.amplitude_rera_count,
        s.proxy_rera_count,
        cell(s.proxy_sensitivity),
        cell(s.proxy_precision),
        cell(s.proxy_f1),
    ])
    downloadCsv(headers, rows, `${fileStem()}.csv`)
}
</script>
