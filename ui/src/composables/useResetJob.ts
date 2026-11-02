import { onBeforeUnmount, onMounted, ref } from 'vue'
import type { Ref } from 'vue'
import type { AxiosError } from 'axios'
import { expectSessionLoss } from '@/api/client'
import { getResetJobStatus } from '@/api/db'
import type { ResetJobEnqueued, ResetJobStatus } from '@/types'

export const POLL_INTERVAL_MS = 2000
export const POLL_FAILURES_BEFORE_WARNING = 30

const ACTIVE_STATES = new Set<ResetJobStatus['state']>(['queued', 'running'])
const JOB_KINDS: ResetJobStatus['kind'][] = ['reset', 'delete_data']

const ALREADY_IN_PROGRESS =
    'Another reset or data deletion is in progress — wait for it to finish and try again.'
const STATUS_UNAVAILABLE =
    'Job status is unavailable because the server restarted or the job expired. Reload this page to check the current state.'
const LOST_CONTACT =
    'Lost contact with the server. The job may still be running on the server. Still trying to reach it.'
const JOB_CANCELLED = 'The server shut down before this job started. Nothing was deleted.'

interface StoredJob {
    token: string
    factoryReset: boolean
}

export interface ResetJobHandle {
    /** True from the moment a job is requested, or from setup when resuming one. */
    running: Ref<boolean>
    /** Whether the current or most recent job deletes accounts; survives a resume. */
    factoryReset: Ref<boolean>
    /** The job's final status: null until it ends, and when its outcome is unknown. */
    status: Ref<ResetJobStatus | null>
    error: Ref<string | null>
    start: (enqueue: () => Promise<ResetJobEnqueued>, factoryReset?: boolean) => Promise<void>
}

// sessionStorage, not localStorage: this token can read a one-time admin invite,
// so it must not outlive the tab. Storage can throw, and losing it only costs the
// resume after a refresh, so every access tolerates failure.
function readStoredJob(key: string): StoredJob | null {
    try {
        const raw = sessionStorage.getItem(key)
        if (!raw) return null
        const { token, factoryReset } = JSON.parse(raw) as Partial<StoredJob>
        return token ? { token, factoryReset: factoryReset === true } : null
    } catch {
        return null
    }
}

function writeStoredJob(key: string, job: StoredJob): void {
    try {
        sessionStorage.setItem(key, JSON.stringify(job))
    } catch {
        // Polling continues from memory.
    }
}

function clearStoredJob(key: string): void {
    try {
        sessionStorage.removeItem(key)
    } catch {
        // Nothing to clear.
    }
}

function storageKeyFor(kind: ResetJobStatus['kind']): string {
    return `snore-reset-job-${kind}`
}

/** Forget every followed job: a token must not carry over to the next sign-in. */
export function clearStoredResetJobs(): void {
    for (const kind of JOB_KINDS) clearStoredJob(storageKeyFor(kind))
}

function terminalError(job: ResetJobStatus): string | null {
    if (job.state === 'failed') return job.error_message ?? 'The job failed.'
    if (job.state === 'cancelled') return JOB_CANCELLED
    return null
}

/**
 * Run a database reset or data deletion job and follow it to its end.
 *
 * Each view that calls this owns its own poll loop. Polling ends only on a terminal
 * state, a 404, or unmount: this poll is the only way a factory reset delivers its
 * invite, so no run of failures may end it. Unmounting keeps the stored token, so
 * returning to the view or refreshing the page resumes the job. `onFinished` runs
 * when the job ends: with its final status, or with null after a 404, when the
 * outcome is unknown.
 */
export function useResetJob(
    kind: ResetJobStatus['kind'],
    onFinished?: (status: ResetJobStatus | null) => void,
): ResetJobHandle {
    const storageKey = storageKeyFor(kind)
    const stored = readStoredJob(storageKey)
    if (stored?.factoryReset) expectSessionLoss(true)

    const running = ref(stored !== null)
    const factoryReset = ref(stored?.factoryReset ?? false)
    const status = ref<ResetJobStatus | null>(null)
    const error = ref<string | null>(null)

    let token = stored?.token ?? ''
    let timer: ReturnType<typeof setTimeout> | null = null
    let failures = 0
    let unmounted = false

    function finish(finalStatus: ResetJobStatus | null, message: string | null): void {
        running.value = false
        status.value = finalStatus
        error.value = message
        // After a factory reset succeeds the session is gone and the invite is on
        // screen, so the expectation lasts until the view unmounts.
        if (!factoryReset.value || finalStatus?.state !== 'succeeded') expectSessionLoss(false)
        onFinished?.(finalStatus)
    }

    function schedulePoll(): void {
        timer = setTimeout(() => void poll(), POLL_INTERVAL_MS)
    }

    function handlePollFailure(err: unknown): void {
        if ((err as AxiosError).response?.status === 404) {
            clearStoredJob(storageKey)
            finish(null, STATUS_UNAVAILABLE)
            return
        }
        failures += 1
        if (failures >= POLL_FAILURES_BEFORE_WARNING) error.value = LOST_CONTACT
        schedulePoll()
    }

    async function poll(): Promise<void> {
        let job: ResetJobStatus
        try {
            job = await getResetJobStatus(token)
        } catch (err: unknown) {
            if (!unmounted) handlePollFailure(err)
            return
        }
        if (unmounted) return
        failures = 0
        error.value = null
        if (ACTIVE_STATES.has(job.state)) {
            schedulePoll()
            return
        }
        clearStoredJob(storageKey)
        finish(job, terminalError(job))
    }

    async function start(
        enqueue: () => Promise<ResetJobEnqueued>,
        isFactoryReset = false,
    ): Promise<void> {
        if (running.value) return
        running.value = true
        factoryReset.value = isFactoryReset
        expectSessionLoss(isFactoryReset)
        status.value = null
        error.value = null
        failures = 0
        try {
            const { job_id } = await enqueue()
            token = job_id
            writeStoredJob(storageKey, { token, factoryReset: isFactoryReset })
            if (!unmounted) schedulePoll()
        } catch (err: unknown) {
            running.value = false
            factoryReset.value = false
            expectSessionLoss(false)
            if ((err as AxiosError).response?.status === 409) {
                error.value = ALREADY_IN_PROGRESS
            } else {
                error.value = err instanceof Error ? err.message : 'Request failed'
            }
        }
    }

    onMounted(() => {
        if (running.value) void poll()
    })

    // Before unmount, not after: a remount in the same patch runs setup first, and
    // this release must not cancel the expectation the new instance just set.
    onBeforeUnmount(() => {
        unmounted = true
        if (timer !== null) clearTimeout(timer)
        expectSessionLoss(false)
    })

    return { running, factoryReset, status, error, start }
}
