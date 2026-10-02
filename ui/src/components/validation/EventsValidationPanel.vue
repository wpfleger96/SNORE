<template>
    <ValidationPanelShell
        validator-type="events"
        :params="{ mode }"
        :load-run-id="loadRunId"
        experimental
        experimental-note="SNORE's event detection is scored against the device's own scored events. Agreement shows how closely the heuristic tracks the device, not clinical accuracy."
        :filename-base="fileStem()"
        @update:report="rawReport = $event"
        @download-csv="onDownloadCsv"
    >
        <template #controls>
            <div class="space-y-1">
                <label class="text-xs font-medium text-muted-foreground">Mode</label>
                <Select v-model="mode">
                    <SelectTrigger class="w-[160px]">
                        <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                        <SelectItem value="aasm">aasm</SelectItem>
                        <SelectItem value="aasm_relaxed">aasm_relaxed</SelectItem>
                        <SelectItem value="resmed">resmed</SelectItem>
                    </SelectContent>
                </Select>
            </div>
        </template>

        <template #default>
            <div v-if="report" class="space-y-6">
                <div class="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
                    <StatCard
                        label="Avg Apnea Sensitivity"
                        :value="ratioPct(report.aggregate.avg_apnea_sensitivity)"
                        unit="%"
                        :decimals="1"
                        field="AggregateMetrics.avg_apnea_sensitivity"
                        glossary-key="sensitivity"
                    />
                    <StatCard
                        label="Avg Apnea F1"
                        :value="ratioPct(report.aggregate.avg_apnea_f1)"
                        unit="%"
                        :decimals="1"
                        field="AggregateMetrics.avg_apnea_f1"
                        glossary-key="f1"
                    />
                    <StatCard
                        label="Avg Hypopnea Sensitivity"
                        :value="ratioPct(report.aggregate.avg_hypopnea_sensitivity)"
                        unit="%"
                        :decimals="1"
                        field="AggregateMetrics.avg_hypopnea_sensitivity"
                        glossary-key="sensitivity"
                    />
                    <StatCard
                        label="Avg Hypopnea F1"
                        :value="ratioPct(report.aggregate.avg_hypopnea_f1)"
                        unit="%"
                        :decimals="1"
                        field="AggregateMetrics.avg_hypopnea_f1"
                        glossary-key="f1"
                    />
                    <StatCard
                        label="Sessions Validated"
                        :value="report.aggregate.total_sessions"
                        :decimals="0"
                    />
                </div>

                <!-- Deliberate mobile treatment: this dense validation metrics matrix stays a horizontally scrolling table rather than cards. -->
                <div class="rounded-md border overflow-x-auto">
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>Date</TableHead>
                                <TableHead class="whitespace-nowrap"
                                    >Duration
                                    <ProvenanceMark
                                        :provenance="
                                            provenanceFor('SessionValidation.duration_hours')
                                        "
                                /></TableHead>
                                <TableHead class="whitespace-nowrap"
                                    >Apnea Sens
                                    <ProvenanceMark
                                        :provenance="
                                            provenanceFor('SessionValidation.apnea_sensitivity')
                                        " />
                                    <InfoHint glossary-key="sensitivity"
                                /></TableHead>
                                <TableHead class="whitespace-nowrap"
                                    >Apnea Prec
                                    <ProvenanceMark
                                        :provenance="
                                            provenanceFor('SessionValidation.apnea_precision')
                                        " />
                                    <InfoHint glossary-key="precision"
                                /></TableHead>
                                <TableHead class="whitespace-nowrap"
                                    >Apnea F1
                                    <ProvenanceMark
                                        :provenance="provenanceFor('SessionValidation.apnea_f1')" />
                                    <InfoHint glossary-key="f1"
                                /></TableHead>
                                <TableHead class="whitespace-nowrap"
                                    >Hypopnea Sens
                                    <ProvenanceMark
                                        :provenance="
                                            provenanceFor('SessionValidation.hypopnea_sensitivity')
                                        " />
                                    <InfoHint glossary-key="sensitivity"
                                /></TableHead>
                                <TableHead class="whitespace-nowrap"
                                    >Hypopnea Prec
                                    <ProvenanceMark
                                        :provenance="
                                            provenanceFor('SessionValidation.hypopnea_precision')
                                        " />
                                    <InfoHint glossary-key="precision"
                                /></TableHead>
                                <TableHead class="whitespace-nowrap"
                                    >Hypopnea F1
                                    <ProvenanceMark
                                        :provenance="
                                            provenanceFor('SessionValidation.hypopnea_f1')
                                        " />
                                    <InfoHint glossary-key="f1"
                                /></TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            <TableRow v-if="report.sessions.length === 0">
                                <TableCell
                                    :colspan="8"
                                    class="py-8 text-center text-muted-foreground"
                                >
                                    No validated sessions found in the selected date range.
                                </TableCell>
                            </TableRow>
                            <TableRow
                                v-for="session in report.sessions"
                                v-else
                                :key="session.session_id"
                                :class="{
                                    'bg-amber-50 dark:bg-amber-950/30': isLowSensitivity(session),
                                    'even:bg-muted/50': !isLowSensitivity(session),
                                }"
                            >
                                <TableCell>
                                    <RouterLink
                                        :to="`/sessions/${session.session_id}/analysis`"
                                        class="text-primary hover:underline"
                                    >
                                        {{ formatDateShort(session.date) }}
                                    </RouterLink>
                                </TableCell>
                                <TableCell>{{ session.duration_hours.toFixed(1) }}h</TableCell>
                                <TableCell>{{ pct(session.apnea_sensitivity) }}</TableCell>
                                <TableCell>{{ pct(session.apnea_precision) }}</TableCell>
                                <TableCell>{{ pct(session.apnea_f1) }}</TableCell>
                                <TableCell>{{ pct(session.hypopnea_sensitivity) }}</TableCell>
                                <TableCell>{{ pct(session.hypopnea_precision) }}</TableCell>
                                <TableCell>{{ pct(session.hypopnea_f1) }}</TableCell>
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
import { RouterLink } from 'vue-router'
import StatCard from '@/components/StatCard.vue'
import InfoHint from '@/components/InfoHint.vue'
import ProvenanceMark from '@/components/ProvenanceMark.vue'
import ValidationPanelShell from '@/components/validation/ValidationPanelShell.vue'
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select'
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table'
import { formatDateShort } from '@/utils/formatting'
import { downloadCsv } from '@/utils/download'
import { provenanceFor } from '@/utils/provenance'
import type { ValidationReport, SessionValidation } from '@/types'

defineProps<{ loadRunId?: number | null }>()

const mode = ref<'aasm' | 'aasm_relaxed' | 'resmed'>('aasm')
const rawReport = ref<Record<string, unknown> | null>(null)
const report = computed(() => rawReport.value as ValidationReport | null)

function ratioPct(value: number | null | undefined): number | null {
    return value != null ? value * 100 : null
}

function pct(value: number | null | undefined): string {
    return value != null ? `${(value * 100).toFixed(1)}%` : '—'
}

function isLowSensitivity(session: SessionValidation): boolean {
    return session.apnea_sensitivity < 0.7 || session.hypopnea_sensitivity < 0.7
}

function fileStem(): string {
    const r = report.value
    return r ? `events-validation-${r.date_range_start}-${r.date_range_end}` : 'events-validation'
}

function onDownloadCsv(): void {
    const r = report.value
    if (!r) return
    const headers = [
        'session_id',
        'date',
        'duration_hours',
        'machine_events',
        'programmatic_events',
        'apnea_sens',
        'apnea_prec',
        'apnea_f1',
        'hypopnea_sens',
        'hypopnea_prec',
        'hypopnea_f1',
        'notes',
    ]
    const rows = r.sessions.map((s) => [
        s.session_id,
        s.date,
        s.duration_hours.toFixed(1),
        s.machine_event_count,
        s.programmatic_event_count,
        `${(s.apnea_sensitivity * 100).toFixed(0)}%`,
        `${(s.apnea_precision * 100).toFixed(0)}%`,
        s.apnea_f1.toFixed(2),
        `${(s.hypopnea_sensitivity * 100).toFixed(0)}%`,
        `${(s.hypopnea_precision * 100).toFixed(0)}%`,
        s.hypopnea_f1.toFixed(2),
        s.notes ?? '',
    ])
    downloadCsv(headers, rows, `${fileStem()}.csv`)
}
</script>
