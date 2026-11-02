import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import type { MockInstance } from 'vitest'
import { enableAutoUnmount, mount } from '@vue/test-utils'
import { AxiosError } from 'axios'
import { defineComponent, h } from 'vue'

const { routerPush } = vi.hoisted(() => ({ routerPush: vi.fn() }))

vi.mock('@/router', () => ({ default: { push: routerPush } }))

import api from '@/api/client'
import { useAuth } from '@/composables/useAuth'
import {
    clearStoredResetJobs,
    POLL_FAILURES_BEFORE_WARNING,
    POLL_INTERVAL_MS,
    useResetJob,
} from '@/composables/useResetJob'
import type { ResetJobEnqueued, ResetJobStatus } from '@/types'

enableAutoUnmount(afterEach)

const TOKEN = 'secret-job-token'
const ENQUEUED: ResetJobEnqueued = { job_id: TOKEN, state: 'queued' }

function jobStatus(overrides: Partial<ResetJobStatus> = {}): ResetJobStatus {
    return {
        kind: 'reset',
        state: 'running',
        error_message: null,
        reset_result: null,
        delete_data_result: null,
        ...overrides,
    }
}

function httpError(status: number) {
    return Object.assign(new Error(`HTTP ${status}`), { response: { status } })
}

function respondWith(status: ResetJobStatus) {
    return { data: status } as never
}

type JobKind = ResetJobStatus['kind']

function mountJob(kind: JobKind = 'reset', onFinished?: (s: ResetJobStatus | null) => void) {
    let job!: ReturnType<typeof useResetJob>
    const wrapper = mount(
        defineComponent({
            setup() {
                job = useResetJob(kind, onFinished)
                return () => null
            },
        }),
    )
    return { job, wrapper }
}

async function startedJob(
    kind: JobKind = 'reset',
    onFinished?: (s: ResetJobStatus | null) => void,
) {
    const mounted = mountJob(kind, onFinished)
    await mounted.job.start(() => Promise.resolve(ENQUEUED))
    return mounted
}

async function pollOnce() {
    await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS)
}

/** A 401 from a request that is not part of the job, through the real interceptor. */
async function unrelatedRequestGets401() {
    await api
        .get('/validation/runs', {
            adapter: (config) => {
                const response = { status: 401, statusText: '', data: {}, headers: {}, config }
                return Promise.reject(new AxiosError('Unauthorized', '', config, null, response))
            },
        })
        .catch(() => undefined)
}

describe('useResetJob', () => {
    let request: MockInstance<typeof api.request>

    beforeEach(() => {
        vi.useFakeTimers()
        sessionStorage.clear()
        routerPush.mockClear()
        request = vi.spyOn(api, 'request').mockResolvedValue(respondWith(jobStatus()))
    })

    afterEach(() => {
        vi.clearAllTimers()
        vi.useRealTimers()
        vi.restoreAllMocks()
    })

    it('test_polls_until_success_then_clears_stored_token', async () => {
        const finished = vi.fn()
        const { job } = await startedJob('reset', finished)

        expect(job.running.value).toBe(true)
        expect(sessionStorage.length).toBe(1)

        await pollOnce()
        expect(request).toHaveBeenCalledTimes(1)
        expect(job.running.value).toBe(true)

        const succeeded = jobStatus({ state: 'succeeded' })
        request.mockResolvedValue(respondWith(succeeded))
        await pollOnce()

        expect(request).toHaveBeenCalledTimes(2)
        expect(job.running.value).toBe(false)
        expect(job.status.value).toEqual(succeeded)
        expect(job.error.value).toBeNull()
        expect(sessionStorage.length).toBe(0)
        expect(finished).toHaveBeenCalledExactlyOnceWith(succeeded)

        await pollOnce()
        expect(request).toHaveBeenCalledTimes(2)
    })

    it('test_token_travels_only_in_the_request_header', async () => {
        await startedJob()
        await pollOnce()

        const config = request.mock.calls[0][0]
        expect(config.url).toBe('/db/reset-jobs/status')
        expect(config.headers).toEqual({ 'X-Reset-Job-Token': TOKEN })
        expect(config.params).toBeUndefined()
    })

    it('test_slow_poll_never_overlaps_the_next_one', async () => {
        await startedJob()

        let respond!: (value: never) => void
        request.mockImplementationOnce(() => new Promise((resolve) => (respond = resolve)))

        await pollOnce()
        expect(request).toHaveBeenCalledTimes(1)

        await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS * 10)
        expect(request).toHaveBeenCalledTimes(1)

        respond(respondWith(jobStatus()))
        await pollOnce()
        expect(request).toHaveBeenCalledTimes(2)
    })

    it('test_start_is_ignored_while_a_job_is_running', async () => {
        const { job } = await startedJob()
        const enqueue = vi.fn(() => Promise.resolve(ENQUEUED))

        await job.start(enqueue)

        expect(enqueue).not.toHaveBeenCalled()
    })

    it('test_remount_resumes_job_with_its_token_and_factory_flag', async () => {
        const first = mountJob()
        await first.job.start(() => Promise.resolve(ENQUEUED), true)
        first.wrapper.unmount()

        const { job } = mountJob()

        expect(job.running.value).toBe(true)
        expect(job.factoryReset.value).toBe(true)

        await vi.advanceTimersByTimeAsync(0)
        expect(request).toHaveBeenCalledTimes(1)
        expect(request.mock.calls[0][0].headers).toEqual({ 'X-Reset-Job-Token': TOKEN })
    })

    it('test_stored_token_of_another_kind_is_not_resumed', async () => {
        const first = await startedJob('delete_data')
        first.wrapper.unmount()

        const { job } = mountJob('reset')

        expect(job.running.value).toBe(false)
        await pollOnce()
        expect(request).not.toHaveBeenCalled()
    })

    it('test_unmount_stops_polling_and_keeps_stored_token', async () => {
        const { wrapper } = await startedJob()

        wrapper.unmount()
        await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS * 5)

        expect(request).not.toHaveBeenCalled()
        expect(sessionStorage.length).toBe(1)
    })

    it('test_response_arriving_after_unmount_is_ignored', async () => {
        const { wrapper } = await startedJob()

        let respond!: (value: never) => void
        request.mockImplementationOnce(() => new Promise((resolve) => (respond = resolve)))
        await pollOnce()

        wrapper.unmount()
        respond(respondWith(jobStatus({ state: 'succeeded' })))
        await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS * 5)

        expect(request).toHaveBeenCalledTimes(1)
        expect(sessionStorage.length).toBe(1)
    })

    it('test_enqueue_finishing_after_unmount_stores_token_without_polling', async () => {
        const { job, wrapper } = mountJob()

        let accept!: (value: ResetJobEnqueued) => void
        const started = job.start(() => new Promise((resolve) => (accept = resolve)))
        wrapper.unmount()
        accept(ENQUEUED)
        await started
        await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS * 5)

        expect(request).not.toHaveBeenCalled()
        expect(sessionStorage.length).toBe(1)
    })

    it('test_unreadable_storage_means_no_resume', () => {
        vi.spyOn(Storage.prototype, 'getItem').mockImplementation(() => {
            throw new DOMException('denied', 'SecurityError')
        })

        const { job } = mountJob()

        expect(job.running.value).toBe(false)
    })

    it.each(['not json', 'null', '{}', '{"token":""}'])(
        'test_malformed_stored_value_%s_means_no_resume',
        (value) => {
            vi.spyOn(Storage.prototype, 'getItem').mockReturnValue(value)

            const { job } = mountJob()

            expect(job.running.value).toBe(false)
        },
    )

    it('test_unwritable_storage_still_follows_job_to_the_end', async () => {
        const denied = () => {
            throw new DOMException('denied', 'QuotaExceededError')
        }
        vi.spyOn(Storage.prototype, 'setItem').mockImplementation(denied)
        vi.spyOn(Storage.prototype, 'removeItem').mockImplementation(denied)
        const { job } = await startedJob()

        request.mockResolvedValue(respondWith(jobStatus({ state: 'succeeded' })))
        await pollOnce()

        expect(job.running.value).toBe(false)
        expect(job.status.value?.state).toBe('succeeded')
    })

    it('test_transient_errors_keep_polling_until_success', async () => {
        const { job } = await startedJob()

        request
            .mockRejectedValueOnce(new Error('Network Error'))
            .mockRejectedValueOnce(httpError(502))
            .mockRejectedValueOnce(httpError(503))
            .mockResolvedValueOnce(respondWith(jobStatus({ state: 'succeeded' })))

        await pollOnce()
        await pollOnce()
        await pollOnce()
        expect(job.running.value).toBe(true)
        expect(job.error.value).toBeNull()

        await pollOnce()
        expect(job.status.value?.state).toBe('succeeded')
        expect(job.error.value).toBeNull()
    })

    it('test_successful_poll_resets_the_failure_count', async () => {
        const { job } = await startedJob()

        for (let round = 0; round < 2; round++) {
            for (let i = 0; i < POLL_FAILURES_BEFORE_WARNING - 1; i++) {
                request.mockRejectedValueOnce(httpError(503))
            }
            request.mockResolvedValueOnce(respondWith(jobStatus()))
        }
        await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS * POLL_FAILURES_BEFORE_WARNING * 2)

        expect(job.running.value).toBe(true)
        expect(job.error.value).toBeNull()
    })

    it('test_outage_warns_keeps_polling_then_warning_clears_and_job_completes', async () => {
        const finished = vi.fn()
        const { job } = await startedJob('reset', finished)
        request.mockRejectedValue(httpError(503))

        await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS * (POLL_FAILURES_BEFORE_WARNING - 1))
        expect(job.error.value).toBeNull()

        await pollOnce()
        expect(job.running.value).toBe(true)
        expect(job.error.value).toContain('may still be running on the server')
        expect(job.error.value).toContain('Still trying to reach it')

        const polledDuringOutage = 100
        await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS * polledDuringOutage)
        expect(request).toHaveBeenCalledTimes(POLL_FAILURES_BEFORE_WARNING + polledDuringOutage)
        expect(job.running.value).toBe(true)
        expect(finished).not.toHaveBeenCalled()
        expect(sessionStorage.length).toBe(1)

        request.mockResolvedValue(respondWith(jobStatus()))
        await pollOnce()
        expect(job.running.value).toBe(true)
        expect(job.error.value).toBeNull()

        const succeeded = jobStatus({ state: 'succeeded' })
        request.mockResolvedValue(respondWith(succeeded))
        await pollOnce()
        expect(job.status.value).toEqual(succeeded)
        expect(job.error.value).toBeNull()
        expect(finished).toHaveBeenCalledExactlyOnceWith(succeeded)
    })

    it('test_slow_failing_polls_never_overlap_during_an_outage', async () => {
        await startedJob()
        const rejections: Array<(reason: Error) => void> = []
        request.mockImplementation(
            () => new Promise((_, reject) => rejections.push(reject)) as never,
        )

        await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS * 10)
        expect(request).toHaveBeenCalledTimes(1)

        rejections[0](new Error('timeout of 15000ms exceeded'))
        await pollOnce()
        expect(request).toHaveBeenCalledTimes(2)
    })

    it('test_404_after_an_outage_replaces_the_warning', async () => {
        const { job } = await startedJob()
        request.mockRejectedValue(httpError(503))
        await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS * POLL_FAILURES_BEFORE_WARNING)

        request.mockRejectedValue(httpError(404))
        await pollOnce()

        expect(job.running.value).toBe(false)
        expect(job.error.value).toContain('server restarted or the job expired')
        expect(sessionStorage.length).toBe(0)
    })

    it('test_status_request_has_a_timeout', async () => {
        await startedJob()
        await pollOnce()

        expect(request.mock.calls[0][0].timeout).toBeGreaterThan(0)
    })

    it('test_404_stops_clears_token_and_reports_status_unavailable', async () => {
        const finished = vi.fn()
        const { job } = await startedJob('reset', finished)
        request.mockRejectedValue(httpError(404))

        await pollOnce()

        expect(job.running.value).toBe(false)
        expect(job.status.value).toBeNull()
        expect(job.error.value).toContain('server restarted or the job expired')
        expect(job.error.value).toContain('current state')
        expect(sessionStorage.length).toBe(0)
        expect(finished).toHaveBeenCalledExactlyOnceWith(null)

        await pollOnce()
        expect(request).toHaveBeenCalledTimes(1)
    })

    it('test_failed_job_surfaces_its_error_message', async () => {
        const { job } = await startedJob()
        request.mockResolvedValue(
            respondWith(jobStatus({ state: 'failed', error_message: 'The deletion failed.' })),
        )

        await pollOnce()

        expect(job.running.value).toBe(false)
        expect(job.error.value).toBe('The deletion failed.')
        expect(sessionStorage.length).toBe(0)
    })

    it('test_cancelled_job_reports_shutdown_and_nothing_deleted', async () => {
        const { job } = await startedJob()
        request.mockResolvedValue(respondWith(jobStatus({ state: 'cancelled' })))

        await pollOnce()

        expect(job.running.value).toBe(false)
        expect(job.error.value).toContain('shut down before')
        expect(job.error.value).toContain('Nothing was deleted')
        expect(sessionStorage.length).toBe(0)
    })

    it('test_conflict_on_start_reports_in_progress_and_does_not_poll', async () => {
        const { job } = mountJob()

        await job.start(() => Promise.reject(httpError(409)), true)
        await pollOnce()

        expect(job.running.value).toBe(false)
        expect(job.factoryReset.value).toBe(false)
        expect(job.error.value).toContain('Another reset or data deletion is in progress')
        expect(request).not.toHaveBeenCalled()
        expect(sessionStorage.length).toBe(0)
    })

    it('test_other_start_failure_surfaces_its_message', async () => {
        const { job } = mountJob()

        await job.start(() => Promise.reject(new Error('Server error')))

        expect(job.running.value).toBe(false)
        expect(job.error.value).toBe('Server error')
    })

    it('test_new_job_clears_the_previous_outcome', async () => {
        const { job } = mountJob()
        await job.start(() => Promise.reject(httpError(409)))

        await job.start(() => Promise.resolve(ENQUEUED))

        expect(job.error.value).toBeNull()
        expect(job.running.value).toBe(true)
    })

    describe('sign-out', () => {
        it('test_clearAuth_removes_stored_token_and_later_mount_does_not_resume', async () => {
            const first = await startedJob()
            first.wrapper.unmount()
            expect(sessionStorage.length).toBe(1)

            useAuth().clearAuth()
            expect(sessionStorage.length).toBe(0)

            const { job } = mountJob()
            await pollOnce()

            expect(job.running.value).toBe(false)
            expect(request).not.toHaveBeenCalled()
        })

        it('test_clearStoredResetJobs_removes_every_kind_and_leaves_other_keys', async () => {
            sessionStorage.setItem('unrelated', 'kept')
            await startedJob('reset')
            await startedJob('delete_data')
            expect(sessionStorage.length).toBe(3)

            clearStoredResetJobs()

            expect(sessionStorage.length).toBe(1)
            expect(sessionStorage.getItem('unrelated')).toBe('kept')
        })

        it('test_clearStoredResetJobs_tolerates_storage_throwing', () => {
            vi.spyOn(Storage.prototype, 'removeItem').mockImplementation(() => {
                throw new DOMException('denied', 'SecurityError')
            })

            expect(() => clearStoredResetJobs()).not.toThrow()
        })

        it('test_unrelated_401_signs_out_and_clears_token_during_a_normal_job', async () => {
            await startedJob()

            await unrelatedRequestGets401()

            expect(sessionStorage.length).toBe(0)
            expect(routerPush).toHaveBeenCalledWith('/')
        })
    })

    describe('factory reset session loss', () => {
        async function startedFactoryReset() {
            const mounted = mountJob()
            await mounted.job.start(() => Promise.resolve(ENQUEUED), true)
            return mounted
        }

        it('test_unrelated_401_while_polling_does_not_sign_out', async () => {
            const { job } = await startedFactoryReset()

            await unrelatedRequestGets401()
            await pollOnce()

            expect(routerPush).not.toHaveBeenCalled()
            expect(sessionStorage.length).toBe(1)
            expect(job.running.value).toBe(true)
            expect(request).toHaveBeenCalledTimes(1)
        })

        it('test_unrelated_401_while_a_resumed_job_polls_does_not_sign_out', async () => {
            const first = await startedFactoryReset()
            first.wrapper.unmount()
            mountJob()

            await unrelatedRequestGets401()

            expect(routerPush).not.toHaveBeenCalled()
            expect(sessionStorage.length).toBe(1)
        })

        it('test_remount_in_one_patch_keeps_expecting_session_loss', async () => {
            const first = await startedFactoryReset()
            first.wrapper.unmount()
            const Child = defineComponent({
                setup() {
                    useResetJob('reset')
                    return () => null
                },
            })
            const Keyed = defineComponent({
                props: { viewKey: { type: Number, required: true } },
                setup: (props) => () => h(Child, { key: props.viewKey }),
            })
            const wrapper = mount(Keyed, { props: { viewKey: 1 } })

            await wrapper.setProps({ viewKey: 2 })
            await unrelatedRequestGets401()

            expect(routerPush).not.toHaveBeenCalled()
        })

        it('test_unrelated_401_after_success_leaves_the_invite_view_in_place', async () => {
            await startedFactoryReset()
            request.mockResolvedValue(respondWith(jobStatus({ state: 'succeeded' })))
            await pollOnce()

            await unrelatedRequestGets401()

            expect(routerPush).not.toHaveBeenCalled()
        })

        it('test_failed_factory_reset_restores_normal_session_expiry', async () => {
            await startedFactoryReset()
            request.mockResolvedValue(respondWith(jobStatus({ state: 'failed' })))
            await pollOnce()

            await unrelatedRequestGets401()

            expect(routerPush).toHaveBeenCalledWith('/')
        })

        it('test_rejected_factory_reset_restores_normal_session_expiry', async () => {
            const { job } = mountJob()
            await job.start(() => Promise.reject(httpError(409)), true)

            await unrelatedRequestGets401()

            expect(routerPush).toHaveBeenCalledWith('/')
        })

        it('test_leaving_the_view_restores_normal_session_expiry', async () => {
            const { wrapper } = await startedFactoryReset()
            wrapper.unmount()

            await unrelatedRequestGets401()

            expect(routerPush).toHaveBeenCalledWith('/')
        })
    })
})
