<template>
    <div class="reports-view">
        <h1 class="page-title">Reports</h1>

        <div class="section-card">
            <h2>Report Type</h2>
            <ToggleGroup
                type="single"
                variant="outline"
                :model-value="reportType"
                @update:model-value="
                    (v) => {
                        if (v) reportType = v as 'summary' | 'comparison'
                    }
                "
            >
                <ToggleGroupItem value="summary">Summary</ToggleGroupItem>
                <ToggleGroupItem value="comparison">Comparison</ToggleGroupItem>
            </ToggleGroup>
        </div>

        <div class="section-card">
            <h2>{{ reportType === 'comparison' ? 'Range A' : 'Date Range' }}</h2>
            <div class="filter-grid">
                <div class="filter-field">
                    <label>From</label>
                    <DatePickerInput
                        v-model="fromA"
                        :is-date-disabled="isDateDisabled"
                        :min-value="minValue"
                        :max-value="maxValue"
                    />
                </div>
                <div class="filter-field">
                    <label>To</label>
                    <DatePickerInput
                        v-model="toA"
                        :is-date-disabled="isDateDisabled"
                        :min-value="minValue"
                        :max-value="maxValue"
                    />
                </div>
            </div>
            <div class="preset-row">
                <Button
                    v-for="p in presets"
                    :key="p.label"
                    variant="outline"
                    size="sm"
                    @click="applyPreset(p.days, 'a')"
                >
                    {{ p.label }}
                </Button>
            </div>
        </div>

        <div v-if="reportType === 'comparison'" class="section-card">
            <h2>Range B</h2>
            <div class="filter-grid">
                <div class="filter-field">
                    <label>From</label>
                    <DatePickerInput
                        v-model="fromB"
                        :is-date-disabled="isDateDisabled"
                        :min-value="minValue"
                        :max-value="maxValue"
                    />
                </div>
                <div class="filter-field">
                    <label>To</label>
                    <DatePickerInput
                        v-model="toB"
                        :is-date-disabled="isDateDisabled"
                        :min-value="minValue"
                        :max-value="maxValue"
                    />
                </div>
            </div>
            <div class="preset-row">
                <Button
                    v-for="p in presets"
                    :key="p.label"
                    variant="outline"
                    size="sm"
                    @click="applyPreset(p.days, 'b')"
                >
                    {{ p.label }}
                </Button>
            </div>
            <div v-if="rxPeriods && rxPeriods.length > 0" class="rx-select">
                <div class="filter-field">
                    <label>Fill from RX Period</label>
                    <Select
                        :model-value="selectedRxPeriod"
                        @update:model-value="(v) => fillRangeB(v as string)"
                    >
                        <SelectTrigger>
                            <SelectValue placeholder="Select RX period…" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem
                                v-for="period in rxPeriods"
                                :key="period.start_date"
                                :value="period.start_date"
                            >
                                {{ period.start_date }} – {{ period.end_date }}
                            </SelectItem>
                        </SelectContent>
                    </Select>
                </div>
            </div>
            <p v-if="rxLoadError" class="rx-note">RX periods unavailable</p>
        </div>

        <div class="export-actions">
            <Button :disabled="generating || !canGenerate" @click="handleGenerate">
                <Loader2 v-if="generating" class="mr-2 h-4 w-4 animate-spin" />
                <Download v-else class="mr-2 h-4 w-4" />
                Generate Report
            </Button>
            <p class="hint-text">The downloaded HTML can be printed to PDF from your browser.</p>
        </div>

        <div v-if="result" class="section-card success-card">
            <Check class="h-5 w-5 success-icon" />
            <div>
                <p class="success-title">Report ready</p>
                <p class="success-size">{{ formatBytes(result.size) }}</p>
            </div>
        </div>

        <ErrorState v-if="error" :message="error ?? ''" />
    </div>
</template>

<script setup lang="ts">
import { computed, onMounted, ref, shallowRef } from 'vue'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { Check, Download, Loader2 } from '@lucide/vue'
import { getSummaryReport, getComparisonReport } from '@/api/reports'
import { downloadBlob } from '@/api/export'
import { getRxHistory } from '@/api/rx'
import { formatBytes, formatIso } from '@/utils/formatting'
import type { RxPeriodResponse } from '@/types'
import { useAvailableDates } from '@/composables/useAvailableDates'
import DatePickerInput from '@/components/DatePickerInput.vue'
import ErrorState from '@/components/ErrorState.vue'

const { load: loadDates, isDateDisabled, minValue, maxValue } = useAvailableDates()

const reportType = ref<'summary' | 'comparison'>('summary')

const fromA = ref('')
const toA = ref('')
const fromB = ref('')
const toB = ref('')

const generating = ref(false)
const result = ref<{ size: number } | null>(null)
const error = ref<string | null>(null)

const selectedRxPeriod = ref('')
const rxPeriods = shallowRef<RxPeriodResponse[] | null>(null)
const rxLoadError = ref(false)

const presets = [
    { label: '30d', days: 30 },
    { label: '90d', days: 90 },
    { label: '180d', days: 180 },
    { label: '1yr', days: 365 },
]

onMounted(async () => {
    void loadDates()
    try {
        rxPeriods.value = await getRxHistory()
    } catch {
        rxLoadError.value = true
    }
})

const canGenerate = computed(() => {
    if (!fromA.value || !toA.value) return false
    if (reportType.value === 'comparison' && (!fromB.value || !toB.value)) return false
    return true
})

function applyPreset(days: number, range: 'a' | 'b'): void {
    const to = new Date()
    const from = new Date()
    from.setDate(from.getDate() - days)
    const toStr = formatIso(to)
    const fromStr = formatIso(from)
    if (range === 'a') {
        fromA.value = fromStr
        toA.value = toStr
    } else {
        fromB.value = fromStr
        toB.value = toStr
    }
}

function fillRangeB(startDate: string): void {
    selectedRxPeriod.value = startDate
    const period = rxPeriods.value?.find((p) => p.start_date === startDate)
    if (period) {
        fromB.value = period.start_date
        toB.value = period.end_date
    }
}

async function handleGenerate(): Promise<void> {
    if (fromA.value && toA.value && fromA.value > toA.value) {
        error.value = 'Range A: From date must be before To date'
        return
    }
    if (reportType.value === 'comparison' && fromB.value && toB.value && fromB.value > toB.value) {
        error.value = 'Range B: From date must be before To date'
        return
    }

    generating.value = true
    error.value = null
    result.value = null

    try {
        let blob: Blob
        let filename: string

        if (reportType.value === 'summary') {
            blob = await getSummaryReport(fromA.value, toA.value)
            filename = `snore-report-summary-${fromA.value}-${toA.value}.html`
        } else {
            blob = await getComparisonReport(fromA.value, toA.value, fromB.value, toB.value)
            filename = `snore-report-comparison-${fromA.value}-${toA.value}-vs-${fromB.value}-${toB.value}.html`
        }

        downloadBlob(blob, filename)
        result.value = { size: blob.size }
    } catch (err: unknown) {
        error.value = err instanceof Error ? err.message : 'Report generation failed'
    } finally {
        generating.value = false
    }
}
</script>

<style scoped>
.reports-view {
    max-width: 800px;
}

.filter-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
    gap: 1rem;
}

.filter-field {
    display: flex;
    flex-direction: column;
    gap: 0.375rem;
}

.filter-field label {
    font-size: 0.875rem;
    font-weight: 500;
    color: var(--color-foreground);
}

.preset-row {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.75rem;
    flex-wrap: wrap;
}

.rx-select {
    margin-top: 0.75rem;
    max-width: 320px;
}

.rx-note {
    font-size: 0.8rem;
    color: var(--color-muted-foreground);
    font-style: italic;
    margin-top: 0.5rem;
}

.export-actions {
    margin-bottom: 1.5rem;
}

.hint-text {
    font-size: 0.875rem;
    color: var(--color-muted-foreground);
    margin-top: 0.5rem;
}

.success-card {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    border-color: var(--color-success);
}

.success-icon {
    color: var(--color-success);
}

.success-title {
    font-weight: 600;
    color: var(--color-success);
}

.success-size {
    font-size: 0.875rem;
    color: var(--color-muted-foreground);
}

@media (max-width: 767.98px) {
    .rx-select {
        max-width: none;
    }
}
</style>
