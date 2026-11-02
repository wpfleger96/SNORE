import { afterEach, describe, expect, it, vi, beforeEach } from 'vitest'
import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { ref } from 'vue'
import { defineComponent } from 'vue'

vi.mock('@/composables/useAuth')
vi.mock('@/api/db')
vi.mock('@/utils/formatting', () => ({ formatDateShort: (d: string) => d }))

// Stub DeleteConfirmDialog: renders the preview slot and a plain button when
// `visible` is true; fires `confirm` on click.  Avoids portal/jsdom
// incompatibilities while still testing the view's wiring (dialog open state,
// confirm handler, etc.).
vi.mock('@/components/DeleteConfirmDialog.vue', () => ({
    default: defineComponent({
        name: 'DeleteConfirmDialog',
        props: ['visible', 'title', 'message', 'loading', 'deleting', 'confirmPhrase'],
        emits: ['update:visible', 'confirm'],
        template: `<div v-if="visible"><slot name="preview" /><button class="stub-confirm" @click="$emit('confirm')">Confirm</button></div>`,
    }),
}))

import { makeAuthMock as baseMakeAuthMock } from './helpers/mockUseAuth'
import DatabaseView from '@/views/DatabaseView.vue'
import { useAuth } from '@/composables/useAuth'
import { getDbStats, getResetJobStatus, resetDb } from '@/api/db'
import { POLL_FAILURES_BEFORE_WARNING, POLL_INTERVAL_MS } from '@/composables/useResetJob'

enableAutoUnmount(afterEach)

const DB_STATS = {
    size_mb: 42.5,
    profile_count: 1,
    device_count: 1,
    session_count: 30,
    day_count: 30,
    event_count: 200,
    waveform_count: 100,
    analysis_count: 25,
    pattern_count: 10,
    sessions_with_waveforms: 28,
    sessions_with_events: 30,
    sessions_with_analysis: 20,
    analyzable_session_count: 28,
    waveform_coverage_pct: 93.3,
    event_coverage_pct: 100.0,
    analysis_coverage_pct: 71.4,
    first_session: '2025-01-01T22:00:00',
    last_session: '2025-08-07T22:00:00',
}

const JOB_ENQUEUED = { job_id: 'job-token', state: 'queued' }

const RESET_RESULT_DATA_ONLY = {
    status: 'success',
    tables_cleared: { sessions: 30, devices: 1 },
    total_rows_deleted: 31,
    size_before_mb: 42.5,
    size_after_mb: 3.2,
    bootstrap_invite_url: null,
}

const RESET_RESULT_FULL = {
    status: 'success',
    tables_cleared: { users: 1, sessions: 30, devices: 1 },
    total_rows_deleted: 32,
    size_before_mb: 42.5,
    size_after_mb: 3.2,
    bootstrap_invite_url: '/invite#abc123',
}

function jobStatus(overrides: Record<string, unknown> = {}) {
    return {
        kind: 'reset',
        state: 'running',
        error_message: null,
        reset_result: null,
        delete_data_result: null,
        ...overrides,
    } as never
}

function httpError(status: number) {
    return Object.assign(new Error(`HTTP ${status}`), { response: { status } })
}

function makeAuthMock(overrides: Record<string, unknown> = {}) {
    vi.mocked(useAuth).mockReturnValue(
        baseMakeAuthMock({
            isLocal: ref(false),
            role: ref('admin'),
            isAuthenticated: ref(true),
            ...overrides,
        }) as never,
    )
}

describe('DatabaseView — danger zone', () => {
    beforeEach(() => {
        // flushPromises schedules through setImmediate, so that stays real.
        vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
        vi.resetAllMocks()
        sessionStorage.clear()
        makeAuthMock()
        vi.mocked(getDbStats).mockResolvedValue(DB_STATS as never)
        vi.mocked(resetDb).mockResolvedValue(JOB_ENQUEUED as never)
        vi.mocked(getResetJobStatus).mockResolvedValue(jobStatus())
    })

    afterEach(() => {
        vi.clearAllTimers()
        vi.useRealTimers()
    })

    async function mountAndLoad() {
        const wrapper = mount(DatabaseView)
        await flushPromises()
        return wrapper
    }

    async function confirmReset(wrapper: Awaited<ReturnType<typeof mountAndLoad>>) {
        await wrapper.find('.danger-card button').trigger('click')
        await wrapper.vm.$nextTick()
        await wrapper.find('.stub-confirm').trigger('click')
        await flushPromises()
    }

    async function confirmFactoryReset(wrapper: Awaited<ReturnType<typeof mountAndLoad>>) {
        await wrapper.find('.include-accounts-checkbox').setValue(true)
        await wrapper.vm.$nextTick()
        await confirmReset(wrapper)
    }

    async function pollOnce() {
        await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS)
    }

    function jobNotice(wrapper: Awaited<ReturnType<typeof mountAndLoad>>) {
        return wrapper.find('[role="status"]')
    }

    function buttonNamed(wrapper: Awaited<ReturnType<typeof mountAndLoad>>, name: string) {
        return wrapper.findAll('button').find((b) => b.text().trim() === name)!
    }

    function isDisabled(button: ReturnType<typeof buttonNamed>) {
        return (button.element as HTMLButtonElement).disabled
    }

    it('test_danger_zone_visible_for_admin', async () => {
        const wrapper = await mountAndLoad()
        expect(wrapper.find('.danger-card').exists()).toBe(true)
    })

    it('test_danger_zone_visible_in_local_mode', async () => {
        makeAuthMock({ isLocal: ref(true), role: ref('admin') })
        const wrapper = await mountAndLoad()
        expect(wrapper.find('.danger-card').exists()).toBe(true)
    })

    it('test_danger_zone_hidden_for_member', async () => {
        makeAuthMock({ role: ref('member') })
        const wrapper = await mountAndLoad()
        expect(wrapper.find('.danger-card').exists()).toBe(false)
    })

    it('test_include_accounts_checkbox_shown_in_multiuser_mode', async () => {
        const wrapper = await mountAndLoad()
        expect(wrapper.find('.include-accounts-checkbox').exists()).toBe(true)
    })

    it('test_include_accounts_checkbox_hidden_in_local_mode', async () => {
        makeAuthMock({ isLocal: ref(true), role: ref('admin') })
        const wrapper = await mountAndLoad()
        expect(wrapper.find('.include-accounts-checkbox').exists()).toBe(false)
    })

    it('test_reset_button_opens_dialog_and_confirm_calls_resetDb', async () => {
        const wrapper = await mountAndLoad()

        // The stub-confirm button only renders when the dialog is open.
        expect(wrapper.find('.stub-confirm').exists()).toBe(false)

        // Click Reset — sets resetDialogOpen = true.
        await wrapper.find('.danger-card button').trigger('click')
        await wrapper.vm.$nextTick()

        // Dialog now open.
        expect(wrapper.find('.stub-confirm').exists()).toBe(true)

        // Fire confirm — triggers handleReset.
        await wrapper.find('.stub-confirm').trigger('click')
        await flushPromises()

        expect(resetDb).toHaveBeenCalledWith({ include_accounts: false })
    })

    it('test_data_only_reset_success_shows_sizes_and_reloads_without_invite_banner', async () => {
        const wrapper = await mountAndLoad()
        await confirmReset(wrapper)

        vi.mocked(getResetJobStatus).mockResolvedValue(
            jobStatus({ state: 'succeeded', reset_result: RESET_RESULT_DATA_ONLY }),
        )
        await pollOnce()
        await flushPromises()

        expect(wrapper.find('.invite-banner').exists()).toBe(false)
        expect(wrapper.find('.danger-card').exists()).toBe(true)
        expect(wrapper.text()).toContain('31 rows deleted')
        expect(wrapper.text()).toContain('42.5 MB → 3.2 MB')
        expect(jobNotice(wrapper).exists()).toBe(false)
        expect(getDbStats).toHaveBeenCalledTimes(2)
    })

    it('test_running_job_disables_reset_and_shows_refresh_safe_notice', async () => {
        const wrapper = await mountAndLoad()
        await confirmReset(wrapper)

        const resetButton = wrapper.find('.danger-card button')
        expect((resetButton.element as HTMLButtonElement).disabled).toBe(true)
        expect(resetButton.find('.animate-spin').exists()).toBe(true)
        expect(jobNotice(wrapper).text()).toContain('several minutes')
        expect(jobNotice(wrapper).text()).toContain('safe to refresh')
    })

    it('test_factory_reset_notice_says_keep_page_open_for_invite', async () => {
        const wrapper = await mountAndLoad()
        await confirmFactoryReset(wrapper)

        expect(jobNotice(wrapper).text()).toContain('Keep this page open')
        expect(jobNotice(wrapper).text()).toContain('admin invite link')
        expect(jobNotice(wrapper).text()).not.toContain('safe to refresh')
    })

    it('test_full_reset_shows_invite_banner_and_never_reloads_stats', async () => {
        const wrapper = await mountAndLoad()
        await confirmFactoryReset(wrapper)
        await pollOnce()
        await pollOnce()

        vi.mocked(getResetJobStatus).mockResolvedValue(
            jobStatus({ state: 'succeeded', reset_result: RESET_RESULT_FULL }),
        )
        await pollOnce()
        await flushPromises()

        expect(wrapper.find('.invite-banner').exists()).toBe(true)
        expect(wrapper.find('.invite-url-text').text()).toContain('/invite#abc123')
        expect(wrapper.find('.error-state').exists()).toBe(false)

        // Only the initial load: a stats call on the dead session would 401, and
        // the axios interceptor's redirect would destroy the banner.
        expect(getDbStats).toHaveBeenCalledTimes(1)
    })

    it('test_factory_reset_locks_session_dependent_actions_until_and_after_invite', async () => {
        const wrapper = await mountAndLoad()
        await confirmFactoryReset(wrapper)

        expect(isDisabled(buttonNamed(wrapper, 'Optimize'))).toBe(true)
        expect(isDisabled(buttonNamed(wrapper, 'Reset'))).toBe(true)

        vi.mocked(getResetJobStatus).mockResolvedValue(
            jobStatus({ state: 'succeeded', reset_result: RESET_RESULT_FULL }),
        )
        await pollOnce()
        await flushPromises()

        expect(isDisabled(buttonNamed(wrapper, 'Optimize'))).toBe(true)
        expect(isDisabled(buttonNamed(wrapper, 'Reset'))).toBe(true)
    })

    it('test_resumed_factory_reset_never_loads_stats_and_still_delivers_invite', async () => {
        const firstVisit = await mountAndLoad()
        await confirmFactoryReset(firstVisit)
        firstVisit.unmount()
        vi.mocked(getDbStats).mockClear()

        const wrapper = await mountAndLoad()

        expect(getResetJobStatus).toHaveBeenCalledWith('job-token')
        expect(wrapper.find('.loading-row').exists()).toBe(false)
        expect(jobNotice(wrapper).text()).toContain('Keep this page open')
        expect(wrapper.find('.danger-card').exists()).toBe(true)

        vi.mocked(getResetJobStatus).mockResolvedValue(
            jobStatus({ state: 'succeeded', reset_result: RESET_RESULT_FULL }),
        )
        await pollOnce()
        await flushPromises()

        expect(wrapper.find('.invite-url-text').text()).toContain('/invite#abc123')
        expect(getDbStats).not.toHaveBeenCalled()
    })

    it('test_resumed_data_only_reset_loads_stats_and_says_refresh_is_safe', async () => {
        const firstVisit = await mountAndLoad()
        await confirmReset(firstVisit)
        firstVisit.unmount()
        vi.mocked(getDbStats).mockClear()

        const wrapper = await mountAndLoad()

        expect(getDbStats).toHaveBeenCalledTimes(1)
        expect(wrapper.find('.overview-card').exists()).toBe(true)
        expect(jobNotice(wrapper).text()).toContain('safe to refresh')
    })

    it('test_failed_resumed_factory_reset_loads_stats_and_shows_error', async () => {
        const firstVisit = await mountAndLoad()
        await confirmFactoryReset(firstVisit)
        firstVisit.unmount()
        vi.mocked(getDbStats).mockClear()
        vi.mocked(getResetJobStatus).mockResolvedValue(
            jobStatus({ state: 'failed', error_message: 'The deletion failed.' }),
        )

        const wrapper = await mountAndLoad()

        // A failed job rolled back, so the session is alive and stats are safe to load.
        expect(getDbStats).toHaveBeenCalledTimes(1)
        expect(wrapper.find('.overview-card').exists()).toBe(true)
        expect(wrapper.find('.error-state').text()).toContain('The deletion failed.')
    })

    it('test_factory_reset_with_unknown_outcome_does_not_reload_stats', async () => {
        const wrapper = await mountAndLoad()
        await confirmFactoryReset(wrapper)

        vi.mocked(getResetJobStatus).mockRejectedValue(httpError(404))
        await pollOnce()
        await flushPromises()

        expect(wrapper.find('.error-state').text()).toContain('server restarted')
        expect(getDbStats).toHaveBeenCalledTimes(1)
    })

    it('test_data_only_reset_with_unknown_outcome_reloads_stats', async () => {
        const wrapper = await mountAndLoad()
        await confirmReset(wrapper)

        vi.mocked(getResetJobStatus).mockRejectedValue(httpError(404))
        await pollOnce()
        await flushPromises()

        expect(wrapper.find('.error-state').text()).toContain('server restarted')
        expect(getDbStats).toHaveBeenCalledTimes(2)
    })

    it('test_factory_reset_survives_an_outage_and_still_delivers_invite', async () => {
        const wrapper = await mountAndLoad()
        await confirmFactoryReset(wrapper)

        vi.mocked(getResetJobStatus).mockRejectedValue(httpError(503))
        await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS * POLL_FAILURES_BEFORE_WARNING)
        await flushPromises()

        expect(wrapper.find('.error-state').text()).toContain('may still be running')
        expect(jobNotice(wrapper).text()).toContain('Keep this page open')
        expect(isDisabled(buttonNamed(wrapper, 'Reset'))).toBe(true)

        vi.mocked(getResetJobStatus).mockResolvedValue(
            jobStatus({ state: 'succeeded', reset_result: RESET_RESULT_FULL }),
        )
        await pollOnce()
        await flushPromises()

        expect(wrapper.find('.error-state').exists()).toBe(false)
        expect(wrapper.find('.invite-url-text').text()).toContain('/invite#abc123')
        expect(getDbStats).toHaveBeenCalledTimes(1)
    })

    it('test_job_failure_shows_its_error_message', async () => {
        const wrapper = await mountAndLoad()
        await confirmReset(wrapper)

        vi.mocked(getResetJobStatus).mockResolvedValue(
            jobStatus({ state: 'failed', error_message: 'The deletion failed.' }),
        )
        await pollOnce()
        await flushPromises()

        expect(wrapper.find('.error-state').text()).toContain('The deletion failed.')
        expect(jobNotice(wrapper).exists()).toBe(false)
    })

    it('test_reset_conflict_shows_in_progress_message_and_does_not_poll', async () => {
        vi.mocked(resetDb).mockRejectedValueOnce(httpError(409))

        const wrapper = await mountAndLoad()
        await confirmReset(wrapper)
        await pollOnce()

        expect(wrapper.find('.error-state').text()).toContain(
            'Another reset or data deletion is in progress',
        )
        expect(getResetJobStatus).not.toHaveBeenCalled()
        expect(isDisabled(buttonNamed(wrapper, 'Reset'))).toBe(false)
    })

    it('test_reset_error_shows_when_no_invite_url', async () => {
        vi.mocked(resetDb).mockRejectedValueOnce(new Error('Server error'))

        const wrapper = await mountAndLoad()
        await confirmReset(wrapper)

        expect(wrapper.find('.error-state').exists()).toBe(true)
        expect(wrapper.find('.error-state').text()).toContain('Server error')
    })

    describe('reset text', () => {
        function resetCardText(wrapper: Awaited<ReturnType<typeof mountAndLoad>>) {
            return wrapper.find('.danger-card .vacuum-subtitle').text()
        }

        function resetDialogMessage(wrapper: Awaited<ReturnType<typeof mountAndLoad>>) {
            return wrapper.findComponent({ name: 'DeleteConfirmDialog' }).props('message') as string
        }

        function expectUnrestorableDataWarning(text: string) {
            expect(text).toMatch(/raw file archive/i)
            expect(text).toContain('Re-importing only restores what is still on the SD card')
            expect(text).toContain('Apple Health data')
            expect(text).toContain('mask log')
            expect(text).toContain('validation runs')
            expect(text).toContain('disabled sessions come back enabled')
        }

        async function openResetDialog(wrapper: Awaited<ReturnType<typeof mountAndLoad>>) {
            await wrapper.find('.danger-card button').trigger('click')
            await wrapper.vm.$nextTick()
        }

        function previewLabels(wrapper: Awaited<ReturnType<typeof mountAndLoad>>) {
            return wrapper.findAll('.reset-preview-row span:first-child').map((s) => s.text())
        }

        it('test_reset_card_local_mode_warns_about_raw_archive_and_unrestorable_data', async () => {
            makeAuthMock({ isLocal: ref(true) })
            const wrapper = await mountAndLoad()

            expectUnrestorableDataWarning(resetCardText(wrapper))
        })

        it('test_reset_card_multiuser_mode_warns_about_raw_archive_and_unrestorable_data', async () => {
            const wrapper = await mountAndLoad()

            expectUnrestorableDataWarning(resetCardText(wrapper))
        })

        it('test_reset_card_local_mode_does_not_say_profiles_are_deleted', async () => {
            makeAuthMock({ isLocal: ref(true) })
            const wrapper = await mountAndLoad()

            const text = resetCardText(wrapper)
            expect(text).not.toMatch(/delet[^.]*profiles/i)
            expect(text).toContain('Profiles and the schema are preserved')
        })

        it('test_data_only_dialog_message_warns_about_raw_archive_and_unrestorable_data', async () => {
            const wrapper = await mountAndLoad()

            const message = resetDialogMessage(wrapper)
            expectUnrestorableDataWarning(message)
            expect(message).toContain('User accounts and profiles are preserved')
            expect(message).toContain('This cannot be undone.')
        })

        it('test_data_only_dialog_message_local_mode_warns_about_raw_archive', async () => {
            makeAuthMock({ isLocal: ref(true) })
            const wrapper = await mountAndLoad()

            expectUnrestorableDataWarning(resetDialogMessage(wrapper))
        })

        it('test_factory_reset_dialog_message_names_raw_archives_and_accounts', async () => {
            const wrapper = await mountAndLoad()

            await wrapper.find('.include-accounts-checkbox').setValue(true)
            await wrapper.vm.$nextTick()

            const message = resetDialogMessage(wrapper)
            expect(message).toMatch(/raw file archives/i)
            expect(message).toContain('all user accounts')
            expect(message).toContain('one-time admin invite URL')
            expect(message).not.toContain('preserved')
        })

        it('test_factory_reset_dialog_message_warns_about_unrestorable_data', async () => {
            const wrapper = await mountAndLoad()

            await wrapper.find('.include-accounts-checkbox').setValue(true)
            await wrapper.vm.$nextTick()

            expectUnrestorableDataWarning(resetDialogMessage(wrapper))
        })

        it('test_reset_preview_data_only_reset_omits_profiles_row', async () => {
            const wrapper = await mountAndLoad()

            await openResetDialog(wrapper)

            expect(previewLabels(wrapper)).toContain('Sessions')
            expect(previewLabels(wrapper)).not.toContain('Profiles')
        })

        it('test_reset_preview_local_mode_omits_profiles_row', async () => {
            makeAuthMock({ isLocal: ref(true) })
            const wrapper = await mountAndLoad()

            await openResetDialog(wrapper)

            expect(previewLabels(wrapper)).toContain('Sessions')
            expect(previewLabels(wrapper)).not.toContain('Profiles')
        })

        it('test_reset_preview_factory_reset_shows_profiles_row', async () => {
            const wrapper = await mountAndLoad()

            await wrapper.find('.include-accounts-checkbox').setValue(true)
            await openResetDialog(wrapper)

            expect(previewLabels(wrapper)).toContain('Profiles')
        })
    })
})
