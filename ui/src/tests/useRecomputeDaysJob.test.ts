import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { defineComponent } from 'vue'

vi.mock('@/api/db')

import { getRecomputeDaysStatus, startRecomputeDays } from '@/api/db'
import { useRecomputeDaysJob } from '@/composables/useRecomputeDaysJob'
import { POLL_FAILURES_BEFORE_WARNING, POLL_INTERVAL_MS } from '@/composables/useResetJob'
import type { RecomputeDaysJobStatus } from '@/types'

enableAutoUnmount(afterEach)

function status(overrides: Partial<RecomputeDaysJobStatus> = {}): RecomputeDaysJobStatus {
    return {
        job_id: 'recompute-1',
        state: 'running',
        days_processed: 0,
        days_total: null,
        error_message: null,
        result: null,
        ...overrides,
    }
}

const SUCCEEDED = status({
    state: 'succeeded',
    days_processed: 3,
    days_total: 3,
    result: { days_recomputed: 3, days_pruned: 0 },
})

// The API client's interceptor copies the server's `detail` into `message`.
function httpError(status: number, message = `HTTP ${status}`) {
    return Object.assign(new Error(message), { response: { status } })
}

function mountJob(resumeOnMount = false) {
    const onSucceeded = vi.fn()
    let job!: ReturnType<typeof useRecomputeDaysJob>
    const wrapper = mount(
        defineComponent({
            setup() {
                job = useRecomputeDaysJob({ resumeOnMount: () => resumeOnMount, onSucceeded })
                return () => null
            },
        }),
    )
    return { job, onSucceeded, wrapper }
}

async function pollOnce() {
    await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS)
}

describe('useRecomputeDaysJob', () => {
    beforeEach(() => {
        vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
        vi.resetAllMocks()
        vi.mocked(startRecomputeDays).mockResolvedValue(status({ state: 'queued' }))
        vi.mocked(getRecomputeDaysStatus).mockResolvedValue(status())
    })

    afterEach(() => {
        vi.clearAllTimers()
        vi.useRealTimers()
    })

    it('test_started_job_is_followed_until_it_succeeds', async () => {
        const { job, onSucceeded } = mountJob()

        await job.start()
        expect(job.active.value).toBe(true)
        await pollOnce()
        expect(job.active.value).toBe(true)

        vi.mocked(getRecomputeDaysStatus).mockResolvedValue(SUCCEEDED)
        await pollOnce()

        expect(job.active.value).toBe(false)
        expect(job.job.value).toEqual(SUCCEEDED)
        expect(onSucceeded).toHaveBeenCalledExactlyOnceWith(SUCCEEDED)
        await pollOnce()
        expect(getRecomputeDaysStatus).toHaveBeenCalledTimes(2)
    })

    it('test_failed_job_reports_server_message', async () => {
        const { job, onSucceeded } = mountJob()
        await job.start()

        vi.mocked(getRecomputeDaysStatus).mockResolvedValue(
            status({ state: 'failed', error_message: 'The day recompute failed.' }),
        )
        await pollOnce()

        expect(job.active.value).toBe(false)
        expect(job.error.value).toBe('The day recompute failed.')
        expect(onSucceeded).not.toHaveBeenCalled()
    })

    it('test_cancelled_job_says_finished_days_are_kept', async () => {
        const { job } = mountJob()
        await job.start()

        vi.mocked(getRecomputeDaysStatus).mockResolvedValue(status({ state: 'cancelled' }))
        await pollOnce()

        expect(job.error.value).toContain('Days already recomputed are kept')
    })

    it('test_conflict_with_another_maintenance_task_keeps_server_message', async () => {
        vi.mocked(startRecomputeDays).mockRejectedValue(
            httpError(409, 'Another database maintenance task is in progress.'),
        )
        vi.mocked(getRecomputeDaysStatus).mockRejectedValue(httpError(404))
        const { job } = mountJob()

        await job.start()
        await pollOnce()

        expect(job.active.value).toBe(false)
        expect(job.error.value).toBe('Another database maintenance task is in progress.')
        expect(getRecomputeDaysStatus).toHaveBeenCalledOnce()
    })

    it('test_conflict_with_recompute_from_another_tab_follows_it', async () => {
        vi.mocked(startRecomputeDays).mockRejectedValue(
            httpError(409, 'Another database maintenance task is in progress.'),
        )
        vi.mocked(getRecomputeDaysStatus).mockResolvedValue(
            status({ job_id: 'other-tab', days_processed: 5, days_total: 10 }),
        )
        const { job, onSucceeded } = mountJob()

        await job.start()

        expect(job.active.value).toBe(true)
        expect(job.job.value?.job_id).toBe('other-tab')
        vi.mocked(getRecomputeDaysStatus).mockResolvedValue(SUCCEEDED)
        await pollOnce()
        expect(onSucceeded).toHaveBeenCalledOnce()
    })

    it('test_other_start_failure_does_not_poll', async () => {
        vi.mocked(startRecomputeDays).mockRejectedValue(new Error('Server error'))
        const { job } = mountJob()

        await job.start()
        await pollOnce()

        expect(job.error.value).toBe('Server error')
        expect(getRecomputeDaysStatus).not.toHaveBeenCalled()
    })

    it('test_repeated_poll_failures_warn_and_keep_polling_until_contact_returns', async () => {
        const { job } = mountJob()
        await job.start()
        vi.mocked(getRecomputeDaysStatus).mockRejectedValue(httpError(502))

        for (let i = 1; i < POLL_FAILURES_BEFORE_WARNING; i++) await pollOnce()
        expect(job.error.value).toBeNull()
        await pollOnce()
        expect(job.error.value).toContain('Lost contact with the server')
        expect(job.active.value).toBe(true)

        vi.mocked(getRecomputeDaysStatus).mockResolvedValue(status())
        await pollOnce()
        expect(job.error.value).toBeNull()
    })

    it('test_forbidden_poll_stops_following', async () => {
        const { job } = mountJob()
        await job.start()
        vi.mocked(getRecomputeDaysStatus).mockRejectedValue(httpError(403, 'Admin only'))

        await pollOnce()
        await pollOnce()

        expect(job.active.value).toBe(false)
        expect(job.error.value).toBe('Admin only')
        expect(getRecomputeDaysStatus).toHaveBeenCalledOnce()
    })

    it('test_404_while_following_reports_status_lost', async () => {
        const { job } = mountJob()
        await job.start()
        vi.mocked(getRecomputeDaysStatus).mockRejectedValue(httpError(404))

        await pollOnce()

        expect(job.active.value).toBe(false)
        expect(job.job.value).toBeNull()
        expect(job.error.value).toContain('server restarted or the job expired')
    })

    it('test_running_job_is_resumed_on_mount_and_followed', async () => {
        const { job, onSucceeded } = mountJob(true)
        await flushPromises()

        expect(job.active.value).toBe(true)
        vi.mocked(getRecomputeDaysStatus).mockResolvedValue(SUCCEEDED)
        await pollOnce()

        expect(onSucceeded).toHaveBeenCalledOnce()
    })

    it('test_job_finished_before_mount_shows_result_without_success_callback', async () => {
        vi.mocked(getRecomputeDaysStatus).mockResolvedValue(SUCCEEDED)

        const { job, onSucceeded } = mountJob(true)
        await flushPromises()

        expect(job.job.value).toEqual(SUCCEEDED)
        expect(job.active.value).toBe(false)
        expect(onSucceeded).not.toHaveBeenCalled()
    })

    it('test_no_job_on_mount_is_silent', async () => {
        vi.mocked(getRecomputeDaysStatus).mockRejectedValue(httpError(404))

        const { job } = mountJob(true)
        await flushPromises()

        expect(job.job.value).toBeNull()
        expect(job.error.value).toBeNull()
    })

    it('test_no_poll_on_mount_when_resume_is_not_allowed', async () => {
        mountJob(false)
        await flushPromises()

        expect(getRecomputeDaysStatus).not.toHaveBeenCalled()
    })

    it('test_mount_poll_answered_after_start_is_ignored', async () => {
        let rejectMountPoll!: (err: unknown) => void
        vi.mocked(getRecomputeDaysStatus).mockReturnValueOnce(
            new Promise((_, reject) => (rejectMountPoll = reject)),
        )
        const { job } = mountJob(true)

        await job.start()
        rejectMountPoll(httpError(404))
        await flushPromises()

        expect(job.active.value).toBe(true)
        expect(job.job.value?.state).toBe('queued')
        expect(job.error.value).toBeNull()
    })

    it('test_unmount_stops_polling', async () => {
        const { job, wrapper } = mountJob()
        await job.start()

        wrapper.unmount()
        await pollOnce()

        expect(getRecomputeDaysStatus).not.toHaveBeenCalled()
    })
})
