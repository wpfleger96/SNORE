<template>
    <div class="session-list">
        <h1 class="page-title">Sessions</h1>

        <!-- Filter Panel -->
        <div class="filter-bar">
            <DatePickerInput
                v-model="fromDate"
                :is-date-disabled="isDateDisabled"
                :min-value="minValue"
                :max-value="maxValue"
            />
            <DatePickerInput
                v-model="toDate"
                :is-date-disabled="isDateDisabled"
                :min-value="minValue"
                :max-value="maxValue"
            />
            <Select v-model="selectedDeviceStr">
                <SelectTrigger class="w-[200px]">
                    <SelectValue placeholder="All devices" />
                </SelectTrigger>
                <SelectContent>
                    <SelectItem value="">All devices</SelectItem>
                    <SelectItem v-for="opt in deviceOptions" :key="opt.value" :value="opt.value">
                        {{ opt.label }}
                    </SelectItem>
                </SelectContent>
            </Select>
            <Toggle v-model:pressed="includeDisabled" variant="outline" size="sm">
                <Eye v-if="includeDisabled" class="mr-2 h-4 w-4" />
                <EyeOff v-else class="mr-2 h-4 w-4" />
                {{ includeDisabled ? 'Show Disabled' : 'Active Only' }}
            </Toggle>
            <Button v-if="hasFilters" variant="outline" size="sm" @click="clearFilters">
                <FilterX class="mr-2 h-4 w-4" />
                Clear
            </Button>
            <Button
                v-if="canWrite && selectedIds.size > 0"
                variant="destructive"
                size="sm"
                @click="confirmBulkDelete"
            >
                <Trash2 class="mr-2 h-4 w-4" />
                Delete {{ selectedIds.size }} Selected
            </Button>
        </div>

        <!-- Mobile: card list -->
        <template v-if="isMobile">
            <div v-if="loading" class="card-list">
                <div v-for="i in 4" :key="'skel-' + i" class="data-card">
                    <Skeleton class="mb-3 h-5 w-40" />
                    <Skeleton class="mb-2 h-4 w-full" />
                    <Skeleton class="mb-2 h-4 w-full" />
                    <Skeleton class="h-4 w-2/3" />
                </div>
            </div>
            <div v-else-if="!sessions.length" class="py-8 text-center text-muted-foreground">
                No sessions found.
            </div>
            <template v-else>
                <div class="mobile-sort-row">
                    <span class="mobile-sort-label">Sort by</span>
                    <Select v-model="mobileSortColumn">
                        <SelectTrigger class="mobile-sort-trigger">
                            <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                            <SelectItem value="date">Date</SelectItem>
                            <SelectItem value="duration">Duration</SelectItem>
                        </SelectContent>
                    </Select>
                    <Button
                        variant="outline"
                        size="icon"
                        class="mobile-sort-dir"
                        :disabled="mobileSortColumn === 'duration'"
                        :aria-label="
                            sortDesc
                                ? 'Sorted descending, switch to ascending'
                                : 'Sorted ascending, switch to descending'
                        "
                        @click="toggleSortDirection"
                    >
                        <ArrowDown v-if="sortDesc" class="h-4 w-4" />
                        <ArrowUp v-else class="h-4 w-4" />
                    </Button>
                </div>
                <label class="select-all-row">
                    <input
                        type="checkbox"
                        :checked="allOnPageSelected"
                        :indeterminate="selectedIds.size > 0 && !allOnPageSelected"
                        class="cursor-pointer"
                        @change="toggleSelectAll"
                    />
                    Select all on page
                </label>
                <div class="card-list">
                    <SessionCard
                        v-for="session in sessions"
                        :key="session.id"
                        :session="session"
                        :selected="selectedIds.has(session.id)"
                        :can-write="canWrite"
                        @toggle-select="toggleSelect(session.id)"
                        @toggle-enabled="toggleEnabled(session)"
                        @events="
                            router.push({ name: 'session-events', params: { id: session.id } })
                        "
                        @delete="confirmDelete(session)"
                    />
                </div>
            </template>
        </template>

        <!-- Desktop: table -->
        <Table v-else class="sessions-table">
            <TableHeader>
                <TableRow>
                    <TableHead style="width: 40px">
                        <input
                            type="checkbox"
                            :checked="allOnPageSelected"
                            :indeterminate="selectedIds.size > 0 && !allOnPageSelected"
                            class="cursor-pointer"
                            @change="toggleSelectAll"
                        />
                    </TableHead>
                    <TableHead
                        style="min-width: 180px"
                        class="cursor-pointer select-none"
                        @click="toggleSort('date')"
                    >
                        <span class="inline-flex items-center gap-1">
                            Date
                            <ArrowDown v-if="sortBy === 'date-desc'" class="h-3 w-3 text-primary" />
                            <ArrowUp
                                v-else-if="sortBy === 'date-asc'"
                                class="h-3 w-3 text-primary"
                            />
                            <ArrowUpDown v-else class="h-3 w-3 opacity-30" />
                        </span>
                    </TableHead>
                    <TableHead
                        style="width: 100px"
                        class="cursor-pointer select-none"
                        @click="toggleSort('duration')"
                    >
                        <span class="inline-flex items-center gap-1">
                            Duration
                            <ArrowUp v-if="sortBy === 'duration'" class="h-3 w-3 text-primary" />
                            <ArrowUpDown v-else class="h-3 w-3 opacity-30" />
                        </span>
                    </TableHead>
                    <TableHead style="width: 80px">AHI</TableHead>
                    <TableHead>Device</TableHead>
                    <TableHead style="width: 90px">Status</TableHead>
                    <TableHead style="width: 180px">Actions</TableHead>
                </TableRow>
            </TableHeader>
            <TableBody>
                <template v-if="loading">
                    <TableRow v-for="i in 8" :key="'skel-' + i">
                        <TableCell><Skeleton class="h-4 w-4" /></TableCell>
                        <TableCell><Skeleton class="h-4 w-36" /></TableCell>
                        <TableCell><Skeleton class="h-4 w-12" /></TableCell>
                        <TableCell><Skeleton class="h-4 w-10" /></TableCell>
                        <TableCell><Skeleton class="h-4 w-32" /></TableCell>
                        <TableCell><Skeleton class="h-5 w-16 rounded-full" /></TableCell>
                        <TableCell><Skeleton class="h-6 w-20" /></TableCell>
                    </TableRow>
                </template>
                <TableRow v-else-if="!sessions.length">
                    <TableCell :colspan="7" class="py-8 text-center text-muted-foreground">
                        No sessions found.
                    </TableCell>
                </TableRow>
                <template v-else>
                    <TableRow v-for="session in sessions" :key="session.id">
                        <TableCell>
                            <input
                                type="checkbox"
                                :checked="selectedIds.has(session.id)"
                                class="cursor-pointer"
                                @change="toggleSelect(session.id)"
                            />
                        </TableCell>
                        <TableCell>
                            <RouterLink
                                :to="{ name: 'session-detail', params: { id: session.id } }"
                                class="text-primary no-underline hover:underline"
                            >
                                <span class="block">{{
                                    formatDateWithWeekday(session.therapy_day)
                                }}</span>
                                <span class="block text-xs text-muted-foreground">{{
                                    formatDateTime(session.start_time)
                                }}</span>
                            </RouterLink>
                        </TableCell>
                        <TableCell>{{ session.duration_hours.toFixed(1) }}h</TableCell>
                        <TableCell>
                            <span :class="ahiClass(session.ahi)">
                                {{ session.ahi?.toFixed(1) ?? '---' }}
                            </span>
                        </TableCell>
                        <TableCell>{{ session.manufacturer }} {{ session.model }}</TableCell>
                        <TableCell>
                            <Badge
                                v-if="session.enabled"
                                class="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                            >
                                Active
                            </Badge>
                            <Badge v-else variant="secondary">Disabled</Badge>
                        </TableCell>
                        <TableCell>
                            <div class="row-actions">
                                <Button
                                    v-if="canWrite"
                                    variant="ghost"
                                    size="icon"
                                    :title="session.enabled ? 'Disable' : 'Enable'"
                                    @click.stop="toggleEnabled(session)"
                                >
                                    <Ban v-if="session.enabled" class="h-4 w-4" />
                                    <Check v-else class="h-4 w-4" />
                                </Button>
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    title="Events"
                                    @click.stop="
                                        router.push({
                                            name: 'session-events',
                                            params: { id: session.id },
                                        })
                                    "
                                >
                                    <BarChart3 class="h-4 w-4" />
                                </Button>
                                <Button
                                    v-if="canWrite"
                                    variant="ghost"
                                    size="icon"
                                    title="Delete"
                                    class="text-destructive hover:text-destructive"
                                    @click.stop="confirmDelete(session)"
                                >
                                    <Trash2 class="h-4 w-4" />
                                </Button>
                            </div>
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

        <DeleteConfirmDialog
            v-model:visible="deleteDialogVisible"
            title="Delete Session"
            :message="deleteMessage"
            :loading="deletePreviewLoading"
            :deleting="deleting"
            @confirm="executeDelete"
        >
            <template v-if="deletePreview" #preview>
                <div class="flex gap-5 text-sm text-muted-foreground">
                    <span>{{ deletePreview.event_count }} events</span>
                    <span>{{ deletePreview.waveform_count }} waveforms</span>
                    <span>{{ deletePreview.stats_count }} stats records</span>
                </div>
            </template>
        </DeleteConfirmDialog>
    </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useRouter, useRoute } from 'vue-router'
import {
    Eye,
    EyeOff,
    Ban,
    Check,
    BarChart3,
    Trash2,
    FilterX,
    AlertTriangle,
    ArrowUp,
    ArrowDown,
    ArrowUpDown,
} from '@lucide/vue'
import { Skeleton } from '@/components/ui/skeleton'
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
import {
    Select,
    SelectContent,
    SelectItem,
    SelectTrigger,
    SelectValue,
} from '@/components/ui/select'
import PaginationBar from '@/components/PaginationBar.vue'
import SessionCard from '@/components/SessionCard.vue'
import DeleteConfirmDialog from '@/components/DeleteConfirmDialog.vue'
import {
    getSessions,
    updateSession,
    deleteSessions,
    getSessionDeletePreview,
    getBulkDeletePreview,
} from '@/api/sessions'
import { getDevices } from '@/api/devices'
import { ahiClass, formatDateTime, formatDateWithWeekday } from '@/utils/formatting'
import type { SessionListItem, DeletePreview, DeviceInfo } from '@/types'
import { useAuth } from '@/composables/useAuth'
import { useIsMobile } from '@/composables/useIsMobile'
import { useAvailableDates } from '@/composables/useAvailableDates'
import DatePickerInput from '@/components/DatePickerInput.vue'

const { canWrite } = useAuth()
const { isMobile } = useIsMobile()
const { load: loadDates, isDateDisabled, minValue, maxValue } = useAvailableDates()

const router = useRouter()
const route = useRoute()
const sessions = ref<SessionListItem[]>([])
const loading = ref(true)
const error = ref<string | null>(null)
const totalRecords = ref(0)
const offset = ref(0)
const pageSize = 25
const selectedIds = ref<Set<number>>(new Set())

// Filters
const fromDate = ref<string>((route.query.from as string) ?? '')
const toDate = ref<string>((route.query.to as string) ?? '')
const selectedDevice = ref<string | null>(null)
const includeDisabled = ref(false)
type SessionSortBy = 'date-asc' | 'date-desc' | 'session-id' | 'duration'
const sortBy = ref<SessionSortBy>('date-desc')
const devices = ref<DeviceInfo[]>([])

// Bridge between null-based selectedDevice ref and string-based Select v-model
const selectedDeviceStr = computed({
    get: () => selectedDevice.value ?? '',
    set: (v: string) => {
        selectedDevice.value = v === '' ? null : v
    },
})

const deviceOptions = computed(() =>
    devices.value.map((d) => ({
        label: `${d.manufacturer} ${d.model}`,
        value: `${d.manufacturer} ${d.model}`,
    })),
)

const hasFilters = computed(
    () => fromDate.value !== '' || toDate.value !== '' || selectedDevice.value != null,
)

const allOnPageSelected = computed(
    () => sessions.value.length > 0 && sessions.value.every((s) => selectedIds.value.has(s.id)),
)

function toggleSelectAll(): void {
    if (allOnPageSelected.value) {
        for (const s of sessions.value) selectedIds.value.delete(s.id)
    } else {
        for (const s of sessions.value) selectedIds.value.add(s.id)
    }
}

function toggleSelect(id: number): void {
    if (selectedIds.value.has(id)) {
        selectedIds.value.delete(id)
    } else {
        selectedIds.value.add(id)
    }
}

// Delete
const deleteDialogVisible = ref(false)
const deletePreviewLoading = ref(false)
const deleting = ref(false)
const deletePreview = ref<DeletePreview | null>(null)
const deleteTargetId = ref<number | null>(null)
const bulkDeleteIds = ref<number[]>([])
const deleteMessage = ref('')

async function fetchPage(newOffset: number): Promise<void> {
    loading.value = true
    error.value = null
    try {
        const result = await getSessions({
            limit: pageSize,
            offset: newOffset,
            sort_by: sortBy.value,
            include_disabled: includeDisabled.value || undefined,
            from_date: fromDate.value || undefined,
            to_date: toDate.value || undefined,
            device: selectedDevice.value ?? undefined,
        })
        sessions.value = result.items
        totalRecords.value = result.total
        offset.value = newOffset
        selectedIds.value.clear()
    } catch (err: unknown) {
        error.value = err instanceof Error ? err.message : 'Failed to load sessions'
    } finally {
        loading.value = false
    }
}

function clearFilters(): void {
    fromDate.value = ''
    toDate.value = ''
    selectedDevice.value = null
}

// Mobile sort control — projects the same sortBy ref the table headers set
const mobileSortColumn = computed({
    get: () => (sortBy.value === 'duration' ? 'duration' : 'date'),
    set: (col: string) => {
        if (col === 'duration') {
            sortBy.value = 'duration'
        } else {
            // Preserve the current direction when re-selecting Date
            sortBy.value = sortDesc.value ? 'date-desc' : 'date-asc'
        }
    },
})
const sortDesc = computed(() => sortBy.value === 'date-desc')

function toggleSortDirection(): void {
    if (sortBy.value === 'date-asc' || sortBy.value === 'date-desc') {
        sortBy.value = sortBy.value === 'date-desc' ? 'date-asc' : 'date-desc'
    }
}

function toggleSort(col: 'date' | 'duration'): void {
    if (col === 'date') {
        sortBy.value = sortBy.value === 'date-desc' ? 'date-asc' : 'date-desc'
    } else {
        sortBy.value = sortBy.value === 'duration' ? 'date-desc' : 'duration'
    }
}

async function toggleEnabled(session: SessionListItem): Promise<void> {
    try {
        await updateSession(session.id, { enabled: !session.enabled })
        void fetchPage(offset.value)
    } catch (err: unknown) {
        error.value = err instanceof Error ? err.message : 'Failed to update session'
    }
}

async function confirmDelete(session: SessionListItem): Promise<void> {
    deleteTargetId.value = session.id
    deleteMessage.value = `Delete session for ${formatDateWithWeekday(session.therapy_day)} (started ${formatDateTime(session.start_time)})? This cannot be undone.`
    deleteDialogVisible.value = true
    deletePreviewLoading.value = true
    deletePreview.value = null
    try {
        deletePreview.value = await getSessionDeletePreview(session.id)
    } catch {
        deleteMessage.value += '\n\nCould not load preview — proceed with caution.'
    } finally {
        deletePreviewLoading.value = false
    }
}

async function confirmBulkDelete(): Promise<void> {
    if (selectedIds.value.size === 0) return
    deleteMessage.value = `Delete ${selectedIds.value.size} selected session(s)? This cannot be undone.`
    bulkDeleteIds.value = [...selectedIds.value]
    deleteDialogVisible.value = true
    deletePreviewLoading.value = true
    deletePreview.value = null
    deleteTargetId.value = null
    try {
        deletePreview.value = await getBulkDeletePreview({ session_ids: bulkDeleteIds.value })
    } catch {
        deleteMessage.value += '\n\nCould not load preview — proceed with caution.'
    } finally {
        deletePreviewLoading.value = false
    }
}

async function executeDelete(): Promise<void> {
    deleting.value = true
    try {
        const ids = deleteTargetId.value != null ? [deleteTargetId.value] : bulkDeleteIds.value
        await deleteSessions({ session_ids: ids })
        deleteDialogVisible.value = false
        selectedIds.value.clear()
        void fetchPage(offset.value)
    } catch (err: unknown) {
        error.value = err instanceof Error ? err.message : 'Failed to delete session(s)'
    } finally {
        deleting.value = false
    }
}

// Re-fetch when filters change
watch([fromDate, toDate, selectedDevice, includeDisabled, sortBy], () => void fetchPage(0))

onMounted(async () => {
    void loadDates()
    await fetchPage(0)
    try {
        devices.value = await getDevices()
    } catch {
        // Device filter unavailable — sessions still load
    }
})
</script>

<style scoped>
.session-list {
    max-width: 1200px;
}

.sessions-table {
    cursor: default;
}

.row-actions {
    display: flex;
    gap: 0.25rem;
}

.mobile-sort-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0 0.25rem 0.75rem;
}

.mobile-sort-label {
    font-size: 0.875rem;
    color: var(--color-muted-foreground);
    white-space: nowrap;
}

.mobile-sort-trigger {
    flex: 1;
    min-height: var(--tap-target);
}

.mobile-sort-dir {
    min-width: var(--tap-target);
    min-height: var(--tap-target);
    flex-shrink: 0;
}

.select-all-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.25rem 0.25rem 0.75rem;
    font-size: 0.875rem;
    color: var(--color-muted-foreground);
    cursor: pointer;
}
</style>
