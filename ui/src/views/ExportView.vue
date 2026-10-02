<template>
    <div class="export-view">
        <h1 class="page-title">Export Data</h1>

        <div class="section-card">
            <h2>Format</h2>
            <ToggleGroup
                type="single"
                variant="outline"
                :model-value="format"
                @update:model-value="
                    (v) => {
                        if (v) format = v as string
                    }
                "
            >
                <ToggleGroupItem value="csv">CSV</ToggleGroupItem>
                <ToggleGroupItem value="json">JSON</ToggleGroupItem>
                <ToggleGroupItem value="raw">Raw</ToggleGroupItem>
            </ToggleGroup>
        </div>

        <div class="section-card">
            <h2>Filters</h2>
            <div class="filter-grid">
                <div class="filter-field">
                    <label>From Date</label>
                    <DatePickerInput
                        v-model="fromDate"
                        :is-date-disabled="isDateDisabled"
                        :min-value="minValue"
                        :max-value="maxValue"
                    />
                </div>
                <div class="filter-field">
                    <label>To Date</label>
                    <DatePickerInput
                        v-model="toDate"
                        :is-date-disabled="isDateDisabled"
                        :min-value="minValue"
                        :max-value="maxValue"
                    />
                </div>
                <div class="filter-field">
                    <label>Device</label>
                    <Select v-model="device">
                        <SelectTrigger>
                            <SelectValue placeholder="All Devices" />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="">All Devices</SelectItem>
                            <SelectItem
                                v-for="d in devices ?? []"
                                :key="`${d.manufacturer} ${d.model}`"
                                :value="`${d.manufacturer} ${d.model}`"
                            >
                                {{ d.manufacturer }} {{ d.model }}
                            </SelectItem>
                        </SelectContent>
                    </Select>
                </div>
            </div>
            <ErrorState v-if="devicesError" :message="devicesError" :retry="reloadDevices" />
        </div>

        <div v-if="format === 'csv' || format === 'raw'" class="section-card">
            <h2>Options</h2>
            <div v-if="format === 'csv'" class="option-row">
                <Toggle
                    :model-value="includeWaveforms"
                    @update:model-value="
                        (v) => {
                            includeWaveforms = !!v
                        }
                    "
                >
                    Include Waveforms
                </Toggle>
                <span class="warning-text">(Large file sizes)</span>
            </div>
            <div v-if="format === 'raw'" class="option-row">
                <Toggle
                    :model-value="trimStr"
                    :disabled="!fromDate || !toDate"
                    @update:model-value="
                        (v) => {
                            trimStr = !!v
                        }
                    "
                >
                    Trim STR.edf to date range
                </Toggle>
            </div>
        </div>

        <div class="export-actions">
            <Button :disabled="exporting" @click="handleExport">
                <Loader2 v-if="exporting" class="mr-2 h-4 w-4 animate-spin" />
                <Download v-else class="mr-2 h-4 w-4" />
                Export
            </Button>
        </div>

        <div v-if="exportResult" class="section-card success-card">
            <Check class="h-5 w-5 success-icon" />
            <div>
                <p class="success-title">Export complete</p>
                <p class="success-size">{{ formatSize(exportResult.size) }}</p>
            </div>
        </div>

        <div v-if="error" class="error-state">
            <AlertTriangle class="h-5 w-5" />
            {{ error }}
        </div>
    </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import { Toggle } from '@/components/ui/toggle'
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select'
import { Button } from '@/components/ui/button'
import { Download, Loader2, AlertTriangle, Check } from '@lucide/vue'
import { getDevices } from '@/api/devices'
import { exportCsv, exportJson, exportRaw, downloadBlob } from '@/api/export'
import type { CsvExportParams, RawExportParams, ExportParams } from '@/api/export'
import { useApiLoad } from '@/composables/useApiLoad'
import { useAvailableDates } from '@/composables/useAvailableDates'
import DatePickerInput from '@/components/DatePickerInput.vue'
import ErrorState from '@/components/ErrorState.vue'

const { load: loadDates, isDateDisabled, minValue, maxValue } = useAvailableDates()

const format = ref('csv')
const fromDate = ref('')
const toDate = ref('')
const device = ref('')
const includeWaveforms = ref(false)
const trimStr = ref(false)
const exporting = ref(false)
const exportResult = ref<{ size: number } | null>(null)
const error = ref<string | null>(null)

const {
    data: devices,
    error: devicesError,
    reload: reloadDevices,
} = useApiLoad(() => getDevices(), 'Failed to load devices')

onMounted(() => {
    void loadDates()
})

async function handleExport(): Promise<void> {
    if (fromDate.value && toDate.value && fromDate.value > toDate.value) {
        error.value = 'From date must be before To date'
        return
    }
    exporting.value = true
    error.value = null
    exportResult.value = null
    try {
        const baseParams: ExportParams = {}
        if (fromDate.value) baseParams.from_date = fromDate.value
        if (toDate.value) baseParams.to_date = toDate.value
        if (device.value) baseParams.device = device.value

        let blob: Blob
        let filename: string

        if (format.value === 'csv') {
            const params: CsvExportParams = { ...baseParams }
            if (includeWaveforms.value) params.include_waveforms = true
            blob = await exportCsv(params)
            filename = 'snore-export.zip'
        } else if (format.value === 'json') {
            blob = await exportJson(baseParams)
            filename = 'snore-export.json'
        } else {
            const params: RawExportParams = { ...baseParams, as_zip: true }
            if (trimStr.value && fromDate.value && toDate.value) params.trim_str = true
            blob = await exportRaw(params)
            filename = 'snore-export.zip'
        }

        downloadBlob(blob, filename)
        exportResult.value = { size: blob.size }
    } catch (err: unknown) {
        error.value = err instanceof Error ? err.message : 'Export failed'
    } finally {
        exporting.value = false
    }
}

function formatSize(bytes: number): string {
    if (bytes < 1024) return `${bytes} B`
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}
</script>

<style scoped>
.export-view {
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

.option-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
}

.warning-text {
    font-size: 0.875rem;
    color: var(--color-muted-foreground);
}

.export-actions {
    margin-bottom: 1.5rem;
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
</style>
