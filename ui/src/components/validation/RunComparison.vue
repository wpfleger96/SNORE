<template>
    <div class="space-y-4">
        <div class="flex flex-wrap items-center gap-2">
            <h2 class="text-lg font-semibold">Compare Runs</h2>
            <Select v-model="type">
                <SelectTrigger class="w-[160px]">
                    <SelectValue />
                </SelectTrigger>
                <SelectContent>
                    <SelectItem v-for="t in VALIDATOR_TYPES" :key="t" :value="t">
                        {{ VALIDATOR_LABELS[t] }}
                    </SelectItem>
                </SelectContent>
            </Select>
        </div>

        <p v-if="runsOfType.length < 2" class="text-sm text-muted-foreground">
            Need at least two succeeded {{ VALIDATOR_LABELS[type] }} runs to compare. Run the same
            validator over different date ranges, engine versions, or parameters.
        </p>

        <template v-else>
            <div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
                <div class="space-y-1">
                    <label class="text-xs font-medium text-muted-foreground"
                        >Run A (baseline)</label
                    >
                    <Select v-model="runAId">
                        <SelectTrigger class="w-full">
                            <SelectValue placeholder="Select run" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem
                                v-for="run in runsOfType"
                                :key="run.run_id"
                                :value="String(run.run_id)"
                            >
                                {{ runOptionLabel(run) }}
                            </SelectItem>
                        </SelectContent>
                    </Select>
                </div>
                <div class="space-y-1">
                    <label class="text-xs font-medium text-muted-foreground">Run B (compare)</label>
                    <Select v-model="runBId">
                        <SelectTrigger class="w-full">
                            <SelectValue placeholder="Select run" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem
                                v-for="run in runsOfType"
                                :key="run.run_id"
                                :value="String(run.run_id)"
                            >
                                {{ runOptionLabel(run) }}
                            </SelectItem>
                        </SelectContent>
                    </Select>
                </div>
            </div>

            <div v-if="runA && runB" class="space-y-4">
                <!-- Engine identity + params, differing components highlighted. -->
                <div class="grid grid-cols-1 gap-3 sm:grid-cols-2">
                    <div class="rounded-md border p-3">
                        <p class="mb-2 text-sm font-medium">
                            Run A · {{ runA.date_from }} → {{ runA.date_to }}
                        </p>
                        <IdentityChips
                            :identity="runA.engine_identity"
                            :params="runA.validator_params"
                            :diff-keys="engineDiff"
                            :param-diff-keys="paramDiff"
                        />
                    </div>
                    <div class="rounded-md border p-3">
                        <p class="mb-2 text-sm font-medium">
                            Run B · {{ runB.date_from }} → {{ runB.date_to }}
                        </p>
                        <IdentityChips
                            :identity="runB.engine_identity"
                            :params="runB.validator_params"
                            :diff-keys="engineDiff"
                            :param-diff-keys="paramDiff"
                        />
                    </div>
                </div>

                <div
                    v-if="engineDiff.size === 0 && paramDiff.size === 0"
                    class="text-sm text-muted-foreground"
                >
                    Both runs share the same engine identity and parameters — differences below are
                    from the date range alone.
                </div>
                <div v-else class="text-sm text-muted-foreground">
                    Differing components:
                    <span class="font-medium text-foreground">{{
                        [...engineDiff, ...paramDiff].join(', ')
                    }}</span>
                </div>

                <div v-if="loading" class="py-6 text-center">
                    <Loader2 class="mx-auto h-6 w-6 animate-spin text-muted-foreground" />
                </div>

                <div v-else-if="aggA && aggB" class="rounded-md border overflow-x-auto">
                    <Table>
                        <TableHeader>
                            <TableRow>
                                <TableHead>Metric</TableHead>
                                <TableHead class="text-right">Run A</TableHead>
                                <TableHead class="text-right">Run B</TableHead>
                                <TableHead class="text-right">Δ</TableHead>
                            </TableRow>
                        </TableHeader>
                        <TableBody>
                            <TableRow
                                v-for="row in metricRows"
                                :key="row.metric.path"
                                class="even:bg-muted/50"
                            >
                                <TableCell class="whitespace-nowrap">
                                    {{ row.metric.label }}
                                    <InfoHint
                                        v-if="row.metric.glossaryKey"
                                        :glossary-key="row.metric.glossaryKey"
                                    />
                                </TableCell>
                                <TableCell class="text-right tabular-nums">
                                    {{ formatMetric(row.delta.a, row.metric.kind) }}
                                </TableCell>
                                <TableCell class="text-right tabular-nums">
                                    {{ formatMetric(row.delta.b, row.metric.kind) }}
                                </TableCell>
                                <TableCell
                                    class="text-right font-medium tabular-nums"
                                    :class="deltaClass(row.delta.direction)"
                                >
                                    {{ formatDelta(row.delta.delta, row.metric.kind) }}
                                </TableCell>
                            </TableRow>
                        </TableBody>
                    </Table>
                </div>
            </div>
        </template>
    </div>
</template>

<script setup lang="ts">
import { ref, computed, watch } from 'vue'
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table'
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select'
import { Loader2 } from '@lucide/vue'
import InfoHint from '@/components/InfoHint.vue'
import IdentityChips from '@/components/validation/IdentityChips.vue'
import { useValidationRuns } from '@/composables/useValidationRuns'
import {
    AGGREGATE_METRICS,
    VALIDATOR_LABELS,
    computeDelta,
    diffKeys,
    formatDelta,
    formatMetric,
    type MetricDelta,
    type MetricDescriptor,
} from '@/utils/validationMetrics'
import { formatRelativeTime } from '@/utils/formatting'
import type { ValidatorType, ValidationRunStatus } from '@/types'

const store = useValidationRuns()
const { runs, getDetail } = store

const VALIDATOR_TYPES = Object.keys(VALIDATOR_LABELS) as ValidatorType[]
const type = ref<ValidatorType>('rera')

const runsOfType = computed(() =>
    runs.value.filter((r) => r.validator_type === type.value && r.state === 'succeeded'),
)

const runAId = ref<string>('')
const runBId = ref<string>('')

const runA = computed<ValidationRunStatus | null>(
    () => runsOfType.value.find((r) => String(r.run_id) === runAId.value) ?? null,
)
const runB = computed<ValidationRunStatus | null>(
    () => runsOfType.value.find((r) => String(r.run_id) === runBId.value) ?? null,
)

// Default the two pickers to the two newest runs when the type changes.
watch(
    runsOfType,
    (list) => {
        if (list.length >= 2) {
            if (!list.some((r) => String(r.run_id) === runAId.value)) {
                runAId.value = String(list[1].run_id)
            }
            if (!list.some((r) => String(r.run_id) === runBId.value)) {
                runBId.value = String(list[0].run_id)
            }
        } else {
            runAId.value = ''
            runBId.value = ''
        }
    },
    { immediate: true },
)

const aggA = ref<Record<string, unknown> | null>(null)
const aggB = ref<Record<string, unknown> | null>(null)
const loading = ref(false)

async function loadAggregate(runId: number | null): Promise<Record<string, unknown> | null> {
    if (runId == null) return null
    const detail = await getDetail(runId)
    const report = detail.report_json as { aggregate?: Record<string, unknown> } | null
    return report?.aggregate ?? null
}

watch(
    [runA, runB],
    async ([a, b]) => {
        if (!a || !b) {
            aggA.value = null
            aggB.value = null
            return
        }
        loading.value = true
        try {
            const [ra, rb] = await Promise.all([loadAggregate(a.run_id), loadAggregate(b.run_id)])
            aggA.value = ra
            aggB.value = rb
        } finally {
            loading.value = false
        }
    },
    { immediate: true },
)

const engineDiff = computed(() =>
    diffKeys(runA.value?.engine_identity, runB.value?.engine_identity),
)
const paramDiff = computed(() =>
    diffKeys(runA.value?.validator_params, runB.value?.validator_params),
)

interface MetricRow {
    metric: MetricDescriptor
    delta: MetricDelta
}
const metricRows = computed<MetricRow[]>(() =>
    AGGREGATE_METRICS[type.value].map((metric) => ({
        metric,
        delta: computeDelta(aggA.value, aggB.value, metric),
    })),
)

function deltaClass(direction: MetricDelta['direction']): string {
    if (direction === 'better') return 'text-green-600 dark:text-green-400'
    if (direction === 'worse') return 'text-destructive'
    return 'text-muted-foreground'
}

function runOptionLabel(run: ValidationRunStatus): string {
    return `#${run.run_id} · ${run.date_from}→${run.date_to} · ${formatRelativeTime(run.created_at)}`
}
</script>
