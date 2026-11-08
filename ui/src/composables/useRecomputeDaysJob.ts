import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import type { ComputedRef, Ref } from 'vue'
import type { AxiosError } from 'axios'
import { getRecomputeDaysStatus, startRecomputeDays } from '@/api/db'
import { POLL_FAILURES_BEFORE_WARNING, POLL_INTERVAL_MS } from '@/composables/useResetJob'
import type { RecomputeDaysJobStatus } from '@/types'

const STATUS_LOST =
    'Recompute status is unavailable because the server restarted or the job expired. Days already recomputed are kept, and running it again is safe.'
const LOST_CONTACT =
    'Lost contact with the server. The recompute may still be running on the server. Still trying to reach it.'
const CANCELLED =
    'The server shut down before the recompute finished. Days already recomputed are kept, and running it again is safe.'

export interface RecomputeDaysJobHandle {
    /** The latest status seen: the job this view follows, or the last one the server ran. */
    job: Ref<RecomputeDaysJobStatus | null>
    /** True from the moment a recompute is requested until the followed job ends. */
    active: ComputedRef<boolean>
    error: Ref<string | null>
    start: () => Promise<void>
}

/**
 * Start the admin day recompute and follow it to its end.
 *
 * A start refused with 409 polls once, so a recompute started elsewhere shows
 * up and is followed.
 *
 * On mount, when `resumeOnMount` allows it, one poll picks up a job started
 * earlier, from this tab or another: a running one is followed, a finished one
 * only shows its result. Only a followed job reports its failure or calls
 * `onSucceeded` when it ends. Polling stops on a terminal state, a 404, a
 * 401/403, or unmount; other failures keep polling and warn after
 * `POLL_FAILURES_BEFORE_WARNING` in a row.
 */
export function useRecomputeDaysJob(options: {
    resumeOnMount: () => boolean
    onSucceeded?: (job: RecomputeDaysJobStatus) => void
}): RecomputeDaysJobHandle {
    const job = ref<RecomputeDaysJobStatus | null>(null)
    const starting = ref(false)
    const error = ref<string | null>(null)

    const active = computed(
        () => starting.value || job.value?.state === 'queued' || job.value?.state === 'running',
    )

    let timer: ReturnType<typeof setTimeout> | null = null
    let following = false
    let failures = 0
    let unmounted = false
    // Bumped by start(): a poll answered after a newer start describes an older
    // state and is dropped.
    let generation = 0

    function schedulePoll(): void {
        timer = setTimeout(() => void poll(), POLL_INTERVAL_MS)
    }

    function stopFollowing(message: string | null): void {
        if (following) error.value = message
        following = false
    }

    function handlePollFailure(err: unknown): void {
        const status = (err as AxiosError).response?.status
        if (status === 404) {
            stopFollowing(STATUS_LOST)
            job.value = null
            return
        }
        if (status === 401 || status === 403) {
            stopFollowing(err instanceof Error ? err.message : null)
            job.value = null
            return
        }
        if (!following) return
        failures += 1
        if (failures >= POLL_FAILURES_BEFORE_WARNING) error.value = LOST_CONTACT
        schedulePoll()
    }

    function finish(finalJob: RecomputeDaysJobStatus): void {
        following = false
        if (finalJob.state === 'failed') {
            error.value = finalJob.error_message ?? 'The day recompute failed.'
        } else if (finalJob.state === 'cancelled') {
            error.value = CANCELLED
        } else {
            options.onSucceeded?.(finalJob)
        }
    }

    async function poll(): Promise<void> {
        timer = null
        const requestedAt = generation
        let latest: RecomputeDaysJobStatus
        try {
            latest = await getRecomputeDaysStatus()
        } catch (err: unknown) {
            if (!unmounted && requestedAt === generation) handlePollFailure(err)
            return
        }
        if (unmounted || requestedAt !== generation) return
        failures = 0
        // Clears a lost-contact warning; a refused start keeps its message.
        if (following) error.value = null
        job.value = latest
        if (latest.state === 'queued' || latest.state === 'running') {
            following = true
            schedulePoll()
        } else if (following) {
            finish(latest)
        }
    }

    async function start(): Promise<void> {
        if (active.value) return
        generation += 1
        if (timer !== null) clearTimeout(timer)
        timer = null
        starting.value = true
        error.value = null
        failures = 0
        let refused = false
        try {
            job.value = await startRecomputeDays()
            following = true
            if (!unmounted) schedulePoll()
        } catch (err: unknown) {
            error.value = err instanceof Error ? err.message : 'Request failed'
            refused = (err as AxiosError).response?.status === 409
        } finally {
            starting.value = false
        }
        // The conflict may be a recompute started from another tab: show and follow it.
        if (refused && !unmounted) await poll()
    }

    onMounted(() => {
        if (options.resumeOnMount()) void poll()
    })

    onBeforeUnmount(() => {
        unmounted = true
        if (timer !== null) clearTimeout(timer)
    })

    return { job, active, error, start }
}
