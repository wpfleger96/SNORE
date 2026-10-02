<template>
    <div class="space-y-4">
        <ExperimentalBanner v-if="experimental" :body="experimentalNote" />

        <div class="flex flex-wrap items-end gap-3">
            <div class="space-y-1">
                <label class="text-xs font-medium text-muted-foreground">From</label>
                <DatePickerInput
                    v-model="fromDate"
                    :is-date-disabled="isDateDisabled"
                    :min-value="minValue"
                    :max-value="maxValue"
                />
            </div>
            <div class="space-y-1">
                <label class="text-xs font-medium text-muted-foreground">To</label>
                <DatePickerInput
                    v-model="toDate"
                    :is-date-disabled="isDateDisabled"
                    :min-value="minValue"
                    :max-value="maxValue"
                />
            </div>
            <slot name="controls" />
            <Button :disabled="!canWrite || !fromDate || !toDate || running" @click="handleRun()">
                <Loader2 v-if="running" class="mr-2 h-4 w-4 animate-spin" />
                Run
            </Button>
        </div>

        <div
            v-if="reusedNotice"
            class="flex flex-wrap items-center gap-2 text-sm text-muted-foreground"
        >
            <RotateCcw class="h-4 w-4" />
            <span>Reused an existing run with matching engine and parameters.</span>
            <Button variant="outline" size="sm" :disabled="running" @click="handleRun(true)">
                Re-run fresh
            </Button>
        </div>

        <div v-if="error" class="flex items-center gap-2 text-sm text-destructive">
            <AlertTriangle class="h-4 w-4" />
            {{ error }}
        </div>

        <ValidationJobsBanner
            v-if="activeTypeRuns.length > 0"
            :runs="activeTypeRuns"
            @cancel="handleCancel"
        />

        <div v-if="reportLoading" class="py-8 text-center">
            <Loader2 class="mx-auto h-6 w-6 animate-spin text-muted-foreground" />
        </div>

        <slot v-else-if="report" />

        <p v-else class="py-8 text-center text-sm text-muted-foreground">
            Run a validation or pick a past run from History to see results here.
        </p>

        <div v-if="report" class="flex flex-wrap gap-2">
            <Button variant="outline" size="sm" @click="downloadReportJson"> Download JSON </Button>
            <Button variant="outline" size="sm" @click="emit('download-csv')">
                Download CSV
            </Button>
        </div>
    </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { Button } from '@/components/ui/button'
import DatePickerInput from '@/components/DatePickerInput.vue'
import ExperimentalBanner from '@/components/ExperimentalBanner.vue'
import ValidationJobsBanner from '@/components/validation/ValidationJobsBanner.vue'
import { Loader2, AlertTriangle, RotateCcw } from '@lucide/vue'
import { useAuth } from '@/composables/useAuth'
import { useAvailableDates } from '@/composables/useAvailableDates'
import { useValidationRuns } from '@/composables/useValidationRuns'
import { downloadJson } from '@/utils/download'
import type { ValidatorType } from '@/types'

const props = defineProps<{
    validatorType: ValidatorType
    params?: Record<string, unknown>
    // External request (from run history) to load a specific run into this panel.
    loadRunId?: number | null
    experimental?: boolean
    experimentalNote?: string
    // Base filename (no extension) for the JSON download this shell owns.
    filenameBase: string
}>()

const emit = defineEmits<{
    'update:report': [value: Record<string, unknown> | null]
    'download-csv': []
}>()

const { canWrite } = useAuth()
const { load: loadDates, isDateDisabled, minValue, maxValue } = useAvailableDates()
const store = useValidationRuns()

const fromDate = ref('')
const toDate = ref('')
const running = ref(false)
const error = ref<string | null>(null)
const reusedNotice = ref(false)

const report = ref<Record<string, unknown> | null>(null)
const loadedRunId = ref<number | null>(null)
const trackedRunId = ref<number | null>(null)
const reportLoading = ref(false)

// Mirror the loaded report to the parent (which renders the typed view + downloads).
// flush:'sync' keeps the parent's typed ref in lock-step with the slot's render,
// so the parent's v-if never lags a tick behind the shell showing the slot.
watch(report, (value) => emit('update:report', value), { flush: 'sync' })

// Only in-flight runs for this validator type: the banner tracks live progress,
// while terminal runs live in the History table (feeding the full list here would
// duplicate History and grow unbounded).
const activeTypeRuns = computed(() => store.runsForType(props.validatorType).filter(store.isActive))

async function loadReport(runId: number): Promise<void> {
    reportLoading.value = true
    error.value = null
    try {
        const detail = await store.getDetail(runId)
        report.value = detail.report_json ?? null
        loadedRunId.value = runId
    } catch (err) {
        error.value = err instanceof Error ? err.message : 'Failed to load run report'
    } finally {
        reportLoading.value = false
    }
}

async function handleRun(force = false): Promise<void> {
    if (fromDate.value && toDate.value && fromDate.value > toDate.value) {
        error.value = 'From date must be before To date'
        return
    }
    running.value = true
    error.value = null
    reusedNotice.value = false
    try {
        const status = await store.enqueue({
            validator_type: props.validatorType,
            from_date: fromDate.value,
            to_date: toDate.value,
            params: props.params,
            force,
        })
        trackedRunId.value = status.run_id
        reusedNotice.value = status.reused
        if (status.state === 'succeeded') {
            await loadReport(status.run_id)
        } else {
            // Background job: clear stale results; the poll watcher loads it on success.
            report.value = null
            loadedRunId.value = null
        }
    } catch (err) {
        error.value = err instanceof Error ? err.message : 'Failed to start run'
    } finally {
        running.value = false
    }
}

async function handleCancel(runId: number): Promise<void> {
    try {
        await store.remove(runId)
    } catch {
        /* already terminal / 409 */
    } finally {
        void store.refresh()
    }
}

// The JSON export is identical across validators, so the shell owns it; panels only
// declare the filename base and keep their per-validator CSV shaping.
function downloadReportJson(): void {
    if (report.value) downloadJson(report.value, `${props.filenameBase}.json`)
}

// When a tracked background run reaches a terminal state, load or surface it.
watch(
    () => store.runs.value,
    (runs) => {
        if (trackedRunId.value == null) return
        const run = runs.find((r) => r.run_id === trackedRunId.value)
        if (!run) return
        if (run.state === 'succeeded' && loadedRunId.value !== run.run_id && !reportLoading.value) {
            void loadReport(run.run_id)
        } else if (run.state === 'failed') {
            error.value = run.error_message ?? 'Validation run failed'
            trackedRunId.value = null
        } else if (run.state === 'cancelled') {
            trackedRunId.value = null
        }
    },
    { deep: true },
)

// Load a run requested from the history table.
watch(
    () => props.loadRunId,
    (runId) => {
        if (runId != null) {
            trackedRunId.value = runId
            reusedNotice.value = false
            void loadReport(runId)
        }
    },
)

onMounted(() => {
    void loadDates()
    void store.refresh()
    if (props.loadRunId != null) void loadReport(props.loadRunId)
})
</script>
