<template>
    <ValidationPanelShell
        validator-type="fl"
        :load-run-id="loadRunId"
        experimental
        experimental-note="SNORE's flow-limitation classes are validated against the device's own FLG signal, which is itself a proprietary index — agreement is a consistency check, not a ground-truth comparison."
        :filename-base="fileStem()"
        @update:report="rawReport = $event"
        @download-csv="onDownloadCsv"
    >
        <template #default>
            <div v-if="report" class="space-y-6">
                <div class="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-4">
                    <StatCard
                        label="Spearman (flattening)"
                        :value="report.aggregate.mean_spearman_flattening_r"
                        :decimals="3"
                        glossary-key="spearman_r"
                    />
                    <StatCard
                        label="Spearman (class weight)"
                        :value="report.aggregate.mean_spearman_class_weight_r"
                        :decimals="3"
                        glossary-key="spearman_r"
                    />
                    <StatCard
                        label="AUC25"
                        :value="report.aggregate.mean_auc_t25"
                        :decimals="3"
                        glossary-key="auc"
                    />
                    <StatCard
                        label="AUC50"
                        :value="report.aggregate.mean_auc_t50"
                        :decimals="3"
                        glossary-key="auc"
                    />
                    <StatCard
                        label="Class AUC25"
                        :value="report.aggregate.mean_auc_class_t25"
                        :decimals="3"
                        glossary-key="auc"
                    />
                    <StatCard
                        label="Class AUC50"
                        :value="report.aggregate.mean_auc_class_t50"
                        :decimals="3"
                        glossary-key="auc"
                    />
                    <StatCard
                        label="Cross-night Spearman"
                        :value="report.aggregate.cross_night_spearman_r"
                        :decimals="3"
                        glossary-key="cross_night_spearman"
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
                                <TableHead>Breaths</TableHead>
                                <TableHead class="whitespace-nowrap"
                                    >Spearman <InfoHint glossary-key="spearman_r"
                                /></TableHead>
                                <TableHead class="whitespace-nowrap"
                                    >AUC25 <InfoHint glossary-key="auc"
                                /></TableHead>
                                <TableHead class="whitespace-nowrap"
                                    >AUC50 <InfoHint glossary-key="auc"
                                /></TableHead>
                                <TableHead class="whitespace-nowrap">SNORE FL 95th</TableHead>
                                <TableHead class="whitespace-nowrap">Device FLG 95th</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            <TableRow v-if="report.sessions.length === 0">
                                <TableCell
                                    :colspan="7"
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
                                <TableCell>{{ s.n_breaths_compared }}</TableCell>
                                <TableCell>{{ num(s.spearman_flattening_r) }}</TableCell>
                                <TableCell>{{ num(s.auc_t25) }}</TableCell>
                                <TableCell>{{ num(s.auc_t50) }}</TableCell>
                                <TableCell>{{ num(s.snore_fl_95th) }}</TableCell>
                                <TableCell>{{ num(s.device_flg_95th) }}</TableCell>
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
import type { FlValidationReport } from '@/types'

defineProps<{ loadRunId?: number | null }>()

const rawReport = ref<Record<string, unknown> | null>(null)
const report = computed(() => rawReport.value as FlValidationReport | null)

function num(v: number | null | undefined): string {
    return v != null ? v.toFixed(3) : '—'
}

function fileStem(): string {
    const r = report.value
    return r ? `fl-validation-${r.date_range_start}-${r.date_range_end}` : 'fl-validation'
}

function onDownloadCsv(): void {
    const r = report.value
    if (!r) return
    const headers = [
        'session_id',
        'date',
        'duration_hours',
        'skipped_reason',
        'n_breaths_compared',
        'n_class_breaths_compared',
        'spearman_flattening_r',
        'spearman_class_weight_r',
        'auc_t25',
        'auc_t50',
        'auc_class_t25',
        'auc_class_t50',
        'snore_fl_95th',
        'device_flg_95th',
    ]
    const cell = (v: number | null | undefined): string => (v != null ? v.toFixed(4) : '')
    const rows = r.sessions.map((s) => [
        s.session_id,
        s.date,
        s.duration_hours.toFixed(1),
        s.skipped_reason ?? '',
        s.n_breaths_compared,
        s.n_class_breaths_compared,
        cell(s.spearman_flattening_r),
        cell(s.spearman_class_weight_r),
        cell(s.auc_t25),
        cell(s.auc_t50),
        cell(s.auc_class_t25),
        cell(s.auc_class_t50),
        cell(s.snore_fl_95th),
        cell(s.device_flg_95th),
    ])
    downloadCsv(headers, rows, `${fileStem()}.csv`)
}
</script>
