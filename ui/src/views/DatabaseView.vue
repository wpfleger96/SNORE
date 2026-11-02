<template>
    <div class="database-view">
        <h1 class="page-title !mb-0">Database</h1>

        <div v-if="loading && !data" class="loading-row">
            <Loader2 class="h-5 w-5 animate-spin text-muted-foreground" />
            <span class="loading-text">Loading…</span>
        </div>

        <div v-else-if="error" class="error-row">
            <AlertTriangle class="h-4 w-4" />
            {{ error }}
        </div>

        <template v-else-if="data">
            <!-- Overview -->
            <section class="section">
                <h2 class="section-heading">Overview</h2>
                <div class="overview-card">
                    <div class="overview-row">
                        <HardDrive class="overview-icon" />
                        <div class="overview-detail">
                            <span class="overview-label">File size</span>
                            <span class="overview-value">{{ data.size_mb.toFixed(1) }} MB</span>
                        </div>
                    </div>
                    <div class="overview-row">
                        <Database class="overview-icon" />
                        <div class="overview-detail">
                            <span class="overview-label">Date range</span>
                            <span class="overview-value">
                                <template v-if="data.first_session && data.last_session">
                                    {{ formatDateShort(data.first_session) }} –
                                    {{ formatDateShort(data.last_session) }}
                                </template>
                                <template v-else>—</template>
                            </span>
                        </div>
                    </div>
                    <div class="overview-row">
                        <div class="overview-detail">
                            <span class="overview-label">Devices</span>
                            <span class="overview-value">{{ data.device_count }}</span>
                        </div>
                    </div>
                    <div class="overview-row">
                        <div class="overview-detail">
                            <span class="overview-label">Profiles</span>
                            <span class="overview-value">{{ data.profile_count }}</span>
                        </div>
                    </div>
                </div>
            </section>

            <!-- Counts -->
            <section class="section">
                <h2 class="section-heading">Record Counts</h2>
                <div class="counts-grid">
                    <StatCard label="Sessions" :value="data.session_count" />
                    <StatCard label="Days" :value="data.day_count" />
                    <StatCard label="Events" :value="data.event_count" />
                    <StatCard label="Waveforms" :value="data.waveform_count" />
                    <StatCard label="Analysis Results" :value="data.analysis_count" />
                    <StatCard label="Patterns" :value="data.pattern_count" />
                </div>
            </section>

            <!-- Coverage -->
            <section class="section">
                <h2 class="section-heading">Session Coverage</h2>
                <div class="coverage-list">
                    <div class="coverage-item">
                        <div class="coverage-header">
                            <span class="coverage-label">Waveforms</span>
                            <Badge variant="secondary">
                                {{ data.waveform_coverage_pct.toFixed(1) }}%
                            </Badge>
                        </div>
                        <div class="coverage-track">
                            <div
                                class="coverage-fill"
                                :style="{ width: data.waveform_coverage_pct + '%' }"
                            />
                        </div>
                    </div>
                    <div class="coverage-item">
                        <div class="coverage-header">
                            <span class="coverage-label">Events</span>
                            <Badge variant="secondary">
                                {{ data.event_coverage_pct.toFixed(1) }}%
                            </Badge>
                        </div>
                        <div class="coverage-track">
                            <div
                                class="coverage-fill"
                                :style="{ width: data.event_coverage_pct + '%' }"
                            />
                        </div>
                    </div>
                    <div class="coverage-item">
                        <div class="coverage-header">
                            <span class="coverage-label">Analysis</span>
                            <Badge variant="secondary">
                                {{ data.analysis_coverage_pct.toFixed(1) }}%
                            </Badge>
                        </div>
                        <div class="coverage-track">
                            <div
                                class="coverage-fill"
                                :style="{ width: data.analysis_coverage_pct + '%' }"
                            />
                        </div>
                        <p class="coverage-caption">
                            {{ data.sessions_with_analysis }} of
                            {{ data.analyzable_session_count }} analyzable sessions
                        </p>
                    </div>
                </div>
            </section>

            <!-- Vacuum -->
            <section class="section">
                <h2 class="section-heading">Maintenance</h2>
                <div class="vacuum-card">
                    <div class="vacuum-description">
                        <p class="vacuum-title">Optimize Database</p>
                        <p class="vacuum-subtitle">
                            Runs VACUUM to reclaim unused space and defragment the database file.
                        </p>
                    </div>
                    <Button
                        variant="outline"
                        :disabled="actionsLocked"
                        @click="vacuumDialogOpen = true"
                    >
                        <Loader2 v-if="vacuuming" class="mr-2 h-4 w-4 animate-spin" />
                        Optimize
                    </Button>
                </div>

                <div v-if="vacuumResult" class="vacuum-result">
                    <span class="vacuum-result-label">Last optimization:</span>
                    <span class="vacuum-result-value">
                        {{ vacuumResult.size_before_mb.toFixed(1) }} MB →
                        {{ vacuumResult.size_after_mb.toFixed(1) }} MB
                        <span class="vacuum-savings">
                            (saved
                            {{
                                (vacuumResult.size_before_mb - vacuumResult.size_after_mb).toFixed(
                                    1,
                                )
                            }}
                            MB)
                        </span>
                    </span>
                </div>

                <div v-if="vacuumError" class="error-state">
                    <AlertTriangle class="h-4 w-4" />
                    {{ vacuumError }}
                </div>
            </section>
        </template>

        <!-- Danger Zone — visible to admins in multiuser mode or in local mode -->
        <section v-if="showResetSection" class="section">
            <h2 class="section-heading">Danger Zone</h2>

            <!-- Post-full-reset: show the bootstrap invite URL prominently -->
            <div v-if="bootstrapInviteUrl" class="invite-banner">
                <p class="invite-banner-title">
                    All accounts have been deleted. Use this link to create a new admin account —
                    save it before leaving this page.
                </p>
                <div class="invite-url-row">
                    <code class="invite-url-text">{{ bootstrapInviteUrl }}</code>
                    <Button variant="outline" size="sm" @click="copyInviteUrl">
                        {{ inviteCopied ? 'Copied!' : 'Copy' }}
                    </Button>
                </div>
            </div>

            <div class="danger-card">
                <div class="vacuum-description">
                    <p class="vacuum-title">Reset Database</p>
                    <p class="vacuum-subtitle">
                        <template v-if="isLocal">
                            Delete all sleep data, devices, and the raw file archive. Profiles and
                            the schema are preserved, so you can re-import data afterward.
                        </template>
                        <template v-else>
                            Delete all sleep data and raw file archives across every user account.
                            Profiles and accounts are preserved by default.
                        </template>
                        {{ RESET_RESTORE_WARNING }}
                    </p>

                    <!-- Multiuser-only: include_accounts checkbox -->
                    <label v-if="!isLocal" class="include-accounts-label">
                        <input
                            v-model="includeAccounts"
                            type="checkbox"
                            class="include-accounts-checkbox"
                        />
                        <span>
                            Also delete all user accounts and invites
                            <span class="include-accounts-warning"
                                >(factory reset — everyone is signed out)</span
                            >
                        </span>
                    </label>
                </div>
                <Button
                    variant="destructive"
                    :disabled="actionsLocked"
                    @click="resetDialogOpen = true"
                >
                    <Loader2 v-if="resetting" class="mr-2 h-4 w-4 animate-spin" />
                    Reset
                </Button>
            </div>

            <div v-if="resetting" class="vacuum-result" role="status">
                <Loader2 class="h-4 w-4 shrink-0 animate-spin text-muted-foreground" />
                <span class="vacuum-result-value">{{ resetProgressNotice }}</span>
            </div>

            <div v-if="resetResult && !bootstrapInviteUrl" class="vacuum-result">
                <span class="vacuum-result-label">Last reset:</span>
                <span class="vacuum-result-value">
                    {{ resetResult.total_rows_deleted.toLocaleString() }} rows deleted,
                    {{ resetResult.size_before_mb.toFixed(1) }} MB →
                    {{ resetResult.size_after_mb.toFixed(1) }} MB
                </span>
            </div>

            <div v-if="resetError" class="error-state">
                <AlertTriangle class="h-4 w-4" />
                {{ resetError }}
            </div>
        </section>

        <!-- Vacuum confirmation dialog -->
        <AlertDialog :open="vacuumDialogOpen" @update:open="vacuumDialogOpen = $event">
            <AlertDialogContent>
                <AlertDialogHeader>
                    <AlertDialogTitle>Optimize Database?</AlertDialogTitle>
                    <AlertDialogDescription>
                        This will run VACUUM on the database. The operation may take a few seconds
                        for large databases and will rebuild the file in place.
                    </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <Button :disabled="vacuuming" @click="handleVacuum">
                        <Loader2 v-if="vacuuming" class="mr-2 h-4 w-4 animate-spin" />
                        Optimize
                    </Button>
                </AlertDialogFooter>
            </AlertDialogContent>
        </AlertDialog>

        <!-- Reset confirmation dialog -->
        <DeleteConfirmDialog
            v-if="showDangerZone"
            v-model:visible="resetDialogOpen"
            title="Reset Database"
            :message="resetDialogMessage"
            :loading="false"
            :deleting="resetting"
            confirm-phrase="reset"
            content-class="max-h-[calc(100dvh-2rem)] overflow-y-auto"
            @confirm="handleReset"
        >
            <template v-if="data" #preview>
                <div class="reset-preview">
                    <div class="reset-preview-row">
                        <span>Sessions</span>
                        <span class="font-medium">{{ data.session_count.toLocaleString() }}</span>
                    </div>
                    <div class="reset-preview-row">
                        <span>Days</span>
                        <span class="font-medium">{{ data.day_count.toLocaleString() }}</span>
                    </div>
                    <div class="reset-preview-row">
                        <span>Events</span>
                        <span class="font-medium">{{ data.event_count.toLocaleString() }}</span>
                    </div>
                    <div class="reset-preview-row">
                        <span>Waveforms</span>
                        <span class="font-medium">{{ data.waveform_count.toLocaleString() }}</span>
                    </div>
                    <div class="reset-preview-row">
                        <span>Analysis Results</span>
                        <span class="font-medium">{{ data.analysis_count.toLocaleString() }}</span>
                    </div>
                    <div class="reset-preview-row">
                        <span>Devices</span>
                        <span class="font-medium">{{ data.device_count }}</span>
                    </div>
                    <div v-if="isFactoryReset" class="reset-preview-row">
                        <span>Profiles</span>
                        <span class="font-medium">{{ data.profile_count }}</span>
                    </div>
                </div>
            </template>
        </DeleteConfirmDialog>
    </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useAuth } from '@/composables/useAuth'

import StatCard from '@/components/StatCard.vue'
import DeleteConfirmDialog from '@/components/DeleteConfirmDialog.vue'
import { Button } from '@/components/ui/button'
import {
    AlertDialog,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import { Badge } from '@/components/ui/badge'
import { Loader2, AlertTriangle, Database, HardDrive } from '@lucide/vue'
import { useApiLoad } from '@/composables/useApiLoad'
import { useResetJob } from '@/composables/useResetJob'
import { getDbStats, vacuumDb, resetDb } from '@/api/db'
import { formatDateShort } from '@/utils/formatting'
import type { VacuumResult, ResetJobStatus } from '@/types'

const {
    running: resetting,
    factoryReset: jobIsFactoryReset,
    status: resetStatus,
    error: resetError,
    start: startResetJob,
} = useResetJob('reset', onResetFinished)

const factoryResetRunning = computed(() => resetting.value && jobIsFactoryReset.value)

// A factory reset kills this session when it commits, and a 401 from getDbStats
// triggers the axios interceptor's redirect, which would destroy the invite banner.
// That applies to a resumed job too, so the view must render without stats.
const { data, loading, error, reload } = useApiLoad(async () =>
    factoryResetRunning.value ? null : getDbStats(),
)
const { isLocal, role } = useAuth()

const showDangerZone = computed(() => isLocal.value || role.value === 'admin')
const showResetSection = computed(
    () =>
        showDangerZone.value &&
        (data.value !== null ||
            resetting.value ||
            resetStatus.value !== null ||
            resetError.value !== null),
)

const vacuumDialogOpen = ref(false)
const vacuuming = ref(false)
const vacuumResult = ref<VacuumResult | null>(null)
const vacuumError = ref<string | null>(null)

async function handleVacuum(): Promise<void> {
    vacuuming.value = true
    vacuumDialogOpen.value = false
    vacuumError.value = null
    try {
        vacuumResult.value = await vacuumDb()
        await reload()
    } catch (err: unknown) {
        vacuumError.value = err instanceof Error ? err.message : 'Vacuum failed'
    } finally {
        vacuuming.value = false
    }
}

const resetDialogOpen = ref(false)
const resetResult = computed(() => resetStatus.value?.reset_result ?? null)
const includeAccounts = ref(false)
const bootstrapInviteUrl = computed(() => resetResult.value?.bootstrap_invite_url ?? null)
const inviteCopied = ref(false)

// Optimize and Reset exclude each other, and neither works once a factory reset has
// ended this session: their 401 would redirect away from the invite banner.
const actionsLocked = computed(
    () => vacuuming.value || resetting.value || bootstrapInviteUrl.value !== null,
)

const RESET_RESTORE_WARNING =
    "Re-importing only restores what is still on the SD card. Apple Health data, the mask log, and validation runs are also deleted and won't come back from the card, and disabled sessions come back enabled."

const isFactoryReset = computed(() => !isLocal.value && includeAccounts.value)

const resetDialogMessage = computed(() => {
    if (isFactoryReset.value) {
        return `This will permanently delete ALL data, raw file archives, AND all user accounts. Every user (including you) will be signed out. A one-time admin invite URL will be returned so you can regain access. ${RESET_RESTORE_WARNING}`
    }
    return `This will permanently delete ALL sleep data and raw file archives. User accounts and profiles are preserved. ${RESET_RESTORE_WARNING} This cannot be undone.`
})

const resetProgressNotice = computed(() =>
    factoryResetRunning.value
        ? 'The factory reset is running on the server and can take several minutes on a large database. Keep this page open: the admin invite link appears here when it finishes, and once your session ends a reload lands on the login page.'
        : 'The reset is running on the server and can take several minutes on a large database. It is safe to refresh this page.',
)

async function handleReset(): Promise<void> {
    resetDialogOpen.value = false
    const body = !isLocal.value ? { include_accounts: includeAccounts.value } : undefined
    await startResetJob(() => resetDb(body), isFactoryReset.value)
}

function onResetFinished(status: ResetJobStatus | null): void {
    // Skip reload() when the session is dead, or may be: a factory reset whose
    // outcome is unknown could have committed.
    const sessionDead = status
        ? Boolean(status.reset_result?.bootstrap_invite_url)
        : jobIsFactoryReset.value
    if (!sessionDead) void reload()
}

function copyInviteUrl(): void {
    if (!bootstrapInviteUrl.value) return
    navigator.clipboard.writeText(bootstrapInviteUrl.value).then(() => {
        inviteCopied.value = true
        setTimeout(() => {
            inviteCopied.value = false
        }, 2000)
    })
}
</script>

<style scoped>
.database-view {
    max-width: 1000px;
    margin: 0 auto;
    padding: 1.5rem;
    display: flex;
    flex-direction: column;
    gap: 2rem;
}

.loading-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 2rem 0;
}

.loading-text {
    font-size: 0.875rem;
    color: var(--color-muted-foreground);
}

.error-row {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
    color: var(--color-destructive);
    padding: 1rem 0;
}

.error-state {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.875rem;
    color: var(--color-destructive);
    padding: 0.5rem 0;
}

.section {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

.section-heading {
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-muted-foreground);
}

.overview-card {
    border: 1px solid var(--color-border);
    border-radius: 8px;
    background: var(--color-card);
    padding: 1rem 1.25rem;
    display: flex;
    flex-wrap: wrap;
    gap: 1.25rem 2.5rem;
}

.overview-row {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}

.overview-icon {
    width: 1.1rem;
    height: 1.1rem;
    color: var(--color-muted-foreground);
    flex-shrink: 0;
}

.overview-detail {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
}

.overview-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--color-muted-foreground);
}

.overview-value {
    font-size: 0.95rem;
    font-weight: 600;
    color: var(--color-foreground);
}

.counts-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 1rem;
}

.coverage-list {
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
}

.coverage-item {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
}

.coverage-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
}

.coverage-label {
    font-size: 0.875rem;
    color: var(--color-foreground);
}

.coverage-track {
    height: 0.4rem;
    border-radius: 9999px;
    background: var(--color-muted);
    overflow: hidden;
}

.coverage-fill {
    height: 100%;
    border-radius: 9999px;
    background: var(--color-primary);
    transition: width 0.3s ease;
}

.coverage-caption {
    font-size: 0.75rem;
    color: var(--color-muted-foreground);
    margin: 0;
}

.vacuum-card {
    border: 1px solid var(--color-border);
    border-radius: 8px;
    background: var(--color-card);
    padding: 1rem 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1rem;
}

.vacuum-description {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

.vacuum-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--color-foreground);
}

.vacuum-subtitle {
    font-size: 0.8rem;
    color: var(--color-muted-foreground);
}

.vacuum-result {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.8rem;
    padding: 0.5rem 0;
}

.vacuum-result-label {
    color: var(--color-muted-foreground);
}

.vacuum-result-value {
    font-weight: 500;
    color: var(--color-foreground);
}

.vacuum-savings {
    color: var(--color-muted-foreground);
    font-weight: 400;
}

.danger-card {
    border: 1px solid var(--color-destructive);
    border-radius: 8px;
    background: var(--color-card);
    padding: 1rem 1.25rem;
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
}

.include-accounts-label {
    display: flex;
    align-items: flex-start;
    gap: 0.5rem;
    margin-top: 0.5rem;
    font-size: 0.8rem;
    cursor: pointer;
    color: var(--color-foreground);
}

.include-accounts-checkbox {
    margin-top: 0.1rem;
    flex-shrink: 0;
    accent-color: var(--color-destructive);
}

.include-accounts-warning {
    color: var(--color-destructive);
    font-weight: 500;
}

.reset-preview {
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    font-size: 0.875rem;
}

.reset-preview-row {
    display: flex;
    justify-content: space-between;
    gap: 2rem;
}

.invite-banner {
    border: 2px solid var(--color-destructive);
    border-radius: 8px;
    background: color-mix(in srgb, var(--color-destructive) 8%, var(--color-card));
    padding: 1rem 1.25rem;
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

.invite-banner-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--color-destructive);
    margin: 0;
}

.invite-url-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    flex-wrap: wrap;
}

.invite-url-text {
    flex: 1;
    font-size: 0.8rem;
    word-break: break-all;
    background: var(--color-muted);
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    color: var(--color-foreground);
}

@media (max-width: 767.98px) {
    /* Stack description above a full-width action button */
    .vacuum-card,
    .danger-card {
        flex-direction: column;
        align-items: stretch;
    }
}
</style>
