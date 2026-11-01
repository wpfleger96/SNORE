<template>
    <div class="analysis-mgmt">
        <h1 class="page-title">Analysis Management</h1>

        <!-- Active analysis jobs banner -->
        <AnalysisJobsBanner
            v-if="activeJobs.length > 0 || recentlyFinishedJobs.length > 0"
            :jobs="[...activeJobs, ...recentlyFinishedJobs]"
            @cancel="handleCancelJob"
        />

        <TherapyAssessmentCard />

        <!-- Filter bar -->
        <div class="filter-bar">
            <DatePickerInput
                v-model="fromDate"
                placeholder="From date"
                :is-date-disabled="isDateDisabled"
                :min-value="minValue"
                :max-value="maxValue"
            />
            <DatePickerInput
                v-model="toDate"
                placeholder="To date"
                :is-date-disabled="isDateDisabled"
                :min-value="minValue"
                :max-value="maxValue"
            />
            <Toggle v-model:pressed="analyzedOnly" variant="outline"> Analyzed Only </Toggle>
            <Button v-if="canWrite" variant="outline" size="sm" @click="batchDialogOpen = true">
                <Play class="mr-2 h-4 w-4" />
                Run Batch
            </Button>
        </div>

        <!-- Sessions (mobile cards) -->
        <div v-if="isMobile">
            <div v-if="loading" class="loading-state">
                <Loader2 class="mx-auto h-6 w-6 animate-spin text-muted-foreground" />
            </div>
            <div v-else-if="!sessions.length" class="py-8 text-center text-muted-foreground">
                No sessions found.
            </div>
            <div v-else class="card-list">
                <div v-for="s in sessions" :key="s.session_id" class="data-card">
                    <div class="data-card-header">
                        <RouterLink
                            :to="{
                                name: 'session-analysis',
                                params: { id: s.session_id },
                            }"
                            class="text-primary hover:underline"
                        >
                            {{ formatDateShort(s.session_date) }}
                        </RouterLink>
                        <Badge
                            v-if="s.has_analysis"
                            class="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                        >
                            Analyzed
                        </Badge>
                        <Badge v-else variant="secondary">Not Analyzed</Badge>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label">Duration</span>
                        <span class="data-card-value">{{
                            s.duration_hours != null ? s.duration_hours.toFixed(1) + 'h' : '---'
                        }}</span>
                    </div>
                    <div v-if="canWrite && s.has_analysis" class="data-card-actions">
                        <Button
                            variant="ghost"
                            size="icon"
                            title="Delete analysis"
                            class="mobile-touch-btn text-destructive hover:text-destructive"
                            @click="confirmDelete(s.session_id)"
                        >
                            <Trash2 class="h-4 w-4" />
                        </Button>
                    </div>
                </div>
            </div>
        </div>

        <!-- Sessions table -->
        <Table v-else>
            <TableHeader>
                <TableRow>
                    <TableHead style="min-width: 160px">Session Date</TableHead>
                    <TableHead style="width: 100px">Duration</TableHead>
                    <TableHead style="width: 120px">Status</TableHead>
                    <TableHead style="width: 100px">Actions</TableHead>
                </TableRow>
            </TableHeader>
            <TableBody>
                <TableRow v-if="loading">
                    <TableCell :colspan="4" class="py-8 text-center">
                        <Loader2 class="mx-auto h-6 w-6 animate-spin text-muted-foreground" />
                    </TableCell>
                </TableRow>
                <TableRow v-else-if="!sessions.length">
                    <TableCell :colspan="4" class="py-8 text-center text-muted-foreground">
                        No sessions found.
                    </TableCell>
                </TableRow>
                <template v-else>
                    <TableRow v-for="s in sessions" :key="s.session_id" class="even:bg-muted/50">
                        <TableCell>
                            <RouterLink
                                :to="{
                                    name: 'session-analysis',
                                    params: { id: s.session_id },
                                }"
                                class="text-primary hover:underline"
                            >
                                {{ formatDateShort(s.session_date) }}
                            </RouterLink>
                        </TableCell>
                        <TableCell>{{
                            s.duration_hours != null ? s.duration_hours.toFixed(1) + 'h' : '---'
                        }}</TableCell>
                        <TableCell>
                            <Badge
                                v-if="s.has_analysis"
                                class="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                            >
                                Analyzed
                            </Badge>
                            <Badge v-else variant="secondary">Not Analyzed</Badge>
                        </TableCell>
                        <TableCell>
                            <Button
                                v-if="canWrite && s.has_analysis"
                                variant="ghost"
                                size="icon"
                                title="Delete analysis"
                                class="text-destructive hover:text-destructive"
                                @click="confirmDelete(s.session_id)"
                            >
                                <Trash2 class="h-4 w-4" />
                            </Button>
                        </TableCell>
                    </TableRow>
                </template>
            </TableBody>
        </Table>

        <PaginationBar
            :offset="offset"
            :page-size="pageSize"
            :total="totalRecords"
            @page="fetchPage"
        />

        <div v-if="error" class="error-state">
            <AlertTriangle class="inline h-4 w-4" /> {{ error }}
        </div>

        <!-- Batch dialog -->
        <AlertDialog :open="batchDialogOpen" @update:open="batchDialogOpen = $event">
            <AlertDialogContent class="max-w-[450px]">
                <AlertDialogHeader>
                    <AlertDialogTitle>Run Batch Analysis</AlertDialogTitle>
                    <AlertDialogDescription as-template>
                        <div class="space-y-4">
                            <div class="grid grid-cols-2 gap-3">
                                <div>
                                    <label class="text-sm font-medium">From</label>
                                    <DatePickerInput
                                        v-model="batchFrom"
                                        class="mt-1 w-full"
                                        :is-date-disabled="isDateDisabled"
                                        :min-value="minValue"
                                        :max-value="maxValue"
                                    />
                                </div>
                                <div>
                                    <label class="text-sm font-medium">To</label>
                                    <DatePickerInput
                                        v-model="batchTo"
                                        class="mt-1 w-full"
                                        :is-date-disabled="isDateDisabled"
                                        :min-value="minValue"
                                        :max-value="maxValue"
                                    />
                                </div>
                            </div>
                            <div>
                                <label class="text-sm font-medium">Mode</label>
                                <ToggleGroup
                                    type="single"
                                    variant="outline"
                                    class="mt-1"
                                    :model-value="batchMode"
                                    @update:model-value="
                                        (v) => {
                                            if (v) batchMode = v as string
                                        }
                                    "
                                >
                                    <ToggleGroupItem value="aasm">AASM</ToggleGroupItem>
                                    <ToggleGroupItem value="resmed">ResMed</ToggleGroupItem>
                                </ToggleGroup>
                            </div>
                        </div>
                    </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <Button
                        variant="outline"
                        :disabled="batchRunning"
                        @click="handleAnalyzeMissing"
                    >
                        <Loader2 v-if="batchRunning" class="mr-2 h-4 w-4 animate-spin" />
                        <Play v-else class="mr-2 h-4 w-4" />
                        Analyze Missing
                    </Button>
                    <Button
                        :disabled="!batchFrom || !batchTo || batchRunning"
                        @click="handleBatchRun"
                    >
                        <Loader2 v-if="batchRunning" class="mr-2 h-4 w-4 animate-spin" />
                        <Play v-else class="mr-2 h-4 w-4" />
                        Run
                    </Button>
                </AlertDialogFooter>
            </AlertDialogContent>
        </AlertDialog>

        <!-- Delete dialog -->
        <DeleteConfirmDialog
            v-model:visible="deleteDialogVisible"
            title="Delete Analysis"
            :message="deleteMessage"
            :loading="deletePreviewLoading"
            :deleting="deleting"
            @confirm="executeDelete"
        >
            <template v-if="deletePreviewData" #preview>
                <div class="flex gap-5 text-sm text-muted-foreground">
                    <span>{{ deletePreviewData.records_to_delete }} records</span>
                    <span>{{ deletePreviewData.patterns_count }} patterns</span>
                </div>
            </template>
        </DeleteConfirmDialog>
    </div>
</template>

<script setup lang="ts">
import { ref, watch, computed, onMounted, onUnmounted } from 'vue'
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Toggle } from '@/components/ui/toggle'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import {
    AlertDialog,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import PaginationBar from '@/components/PaginationBar.vue'
import DeleteConfirmDialog from '@/components/DeleteConfirmDialog.vue'
import AnalysisJobsBanner from '@/components/AnalysisJobsBanner.vue'
import DatePickerInput from '@/components/DatePickerInput.vue'
import TherapyAssessmentCard from '@/components/TherapyAssessmentCard.vue'
import { Loader2, AlertTriangle, Play, Trash2 } from '@lucide/vue'
import {
    getAnalysisSessions,
    runBatchAnalysis,
    deleteAnalysis,
    getAnalysisDeletePreview,
    getAnalysisJobs,
    cancelAnalysisJob,
} from '@/api/analysis'
import { formatDateShort } from '@/utils/formatting'
import type { AnalysisListItem, AnalysisDeletePreview, AnalysisJobStatus } from '@/types'
import { useAuth } from '@/composables/useAuth'
import { useAvailableDates } from '@/composables/useAvailableDates'
import { useIsMobile } from '@/composables/useIsMobile'

const { canWrite } = useAuth()
const { isMobile } = useIsMobile()
const { load: loadDates, isDateDisabled, minValue, maxValue } = useAvailableDates()

const sessions = ref<AnalysisListItem[]>([])
const loading = ref(true)
const error = ref<string | null>(null)
const totalRecords = ref(0)
const offset = ref(0)
const pageSize = 25

const fromDate = ref('')
const toDate = ref('')
const analyzedOnly = ref(false)

const batchDialogOpen = ref(false)
const batchFrom = ref('')
const batchTo = ref('')
const batchMode = ref('aasm')
const batchRunning = ref(false)

const analysisJobs = ref<AnalysisJobStatus[]>([])
let pollTimer: ReturnType<typeof setTimeout> | null = null
let stopped = false

const activeJobs = computed(() =>
    analysisJobs.value.filter((j) => j.state === 'queued' || j.state === 'running'),
)
const recentlyFinishedJobs = computed(() =>
    analysisJobs.value.filter(
        (j) => j.state === 'succeeded' || j.state === 'failed' || j.state === 'cancelled',
    ),
)

const deleteDialogVisible = ref(false)
const deletePreviewLoading = ref(false)
const deleting = ref(false)
const deletePreviewData = ref<AnalysisDeletePreview | null>(null)
const deleteTargetId = ref<number | null>(null)
const deleteMessage = ref('')

async function fetchPage(newOffset: number): Promise<void> {
    loading.value = true
    error.value = null
    try {
        const result = await getAnalysisSessions({
            limit: pageSize,
            offset: newOffset,
            from_date: fromDate.value || undefined,
            to_date: toDate.value || undefined,
            analyzed_only: analyzedOnly.value || undefined,
            sort_by: 'date-desc',
        })
        sessions.value = result.items
        totalRecords.value = result.total
        offset.value = newOffset
    } catch (err: unknown) {
        error.value = err instanceof Error ? err.message : 'Failed to load sessions'
    } finally {
        loading.value = false
    }
}

function schedulePoll() {
    if (stopped || pollTimer !== null) return
    pollTimer = setTimeout(async () => {
        pollTimer = null
        if (stopped) return
        await fetchJobs()
    }, 3000)
}

async function fetchJobs() {
    try {
        const { jobs } = await getAnalysisJobs()
        if (stopped) return
        const hadActive = activeJobs.value.length > 0
        analysisJobs.value = jobs
        const hasActive = jobs.some((j) => j.state === 'queued' || j.state === 'running')
        if (hasActive) {
            schedulePoll()
        } else if (hadActive) {
            void fetchPage(0)
        }
    } catch {
        // Ignore poll errors silently
    }
}

async function handleCancelJob(jobId: string) {
    try {
        await cancelAnalysisJob(jobId)
    } catch {
        /* job may already be terminal or returned 409 */
    } finally {
        void fetchJobs()
    }
}

async function handleBatchRun(): Promise<void> {
    if (batchFrom.value && batchTo.value && batchFrom.value > batchTo.value) {
        error.value = 'From date must be before To date'
        batchDialogOpen.value = false
        return
    }
    batchRunning.value = true
    try {
        await runBatchAnalysis({
            from_date: batchFrom.value,
            to_date: batchTo.value,
            modes: [batchMode.value],
            store_results: true,
        })
        batchDialogOpen.value = false
        void fetchJobs()
    } catch (err: unknown) {
        batchDialogOpen.value = false
        error.value = err instanceof Error ? err.message : 'Failed to queue batch analysis'
    } finally {
        batchRunning.value = false
    }
}

async function handleAnalyzeMissing(): Promise<void> {
    batchRunning.value = true
    try {
        await runBatchAnalysis({
            missing_only: true,
            modes: [batchMode.value],
            store_results: true,
        })
        batchDialogOpen.value = false
        void fetchJobs()
    } catch (err: unknown) {
        batchDialogOpen.value = false
        error.value = err instanceof Error ? err.message : 'Failed to queue batch analysis'
    } finally {
        batchRunning.value = false
    }
}

async function confirmDelete(sessionId: number): Promise<void> {
    deleteTargetId.value = sessionId
    deleteMessage.value = 'Delete analysis for this session? This cannot be undone.'
    deleteDialogVisible.value = true
    deletePreviewLoading.value = true
    deletePreviewData.value = null
    try {
        deletePreviewData.value = await getAnalysisDeletePreview({
            session_ids: [sessionId],
        })
    } catch {
        deleteMessage.value += '\n\nCould not load preview — proceed with caution.'
    } finally {
        deletePreviewLoading.value = false
    }
}

async function executeDelete(): Promise<void> {
    if (deleteTargetId.value == null) return
    deleting.value = true
    try {
        await deleteAnalysis({ session_ids: [deleteTargetId.value] })
        deleteDialogVisible.value = false
        void fetchPage(offset.value)
    } catch (err: unknown) {
        error.value = err instanceof Error ? err.message : 'Failed to delete analysis'
    } finally {
        deleting.value = false
    }
}

watch([fromDate, toDate, analyzedOnly], () => void fetchPage(0))

onMounted(() => {
    void loadDates()
    void fetchJobs()
    void fetchPage(0)
})

onUnmounted(() => {
    stopped = true
    if (pollTimer !== null) {
        clearTimeout(pollTimer)
        pollTimer = null
    }
})
</script>

<style scoped>
.analysis-mgmt {
    max-width: 1200px;
}

.mobile-touch-btn {
    min-height: var(--tap-target);
    min-width: var(--tap-target);
}
</style>
