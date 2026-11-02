import { afterEach, describe, expect, it, vi, beforeEach } from 'vitest'
import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'
import { ref, defineComponent } from 'vue'

vi.mock('@/composables/useAuth')
vi.mock('@/composables/useDateFormat')
vi.mock('vue-router')

// Stub DeleteConfirmDialog to avoid portal/jsdom incompatibilities.
vi.mock('@/components/DeleteConfirmDialog.vue', () => ({
    default: defineComponent({
        name: 'DeleteConfirmDialog',
        props: ['visible', 'title', 'message', 'loading', 'deleting', 'confirmPhrase'],
        emits: ['update:visible', 'confirm'],
        template: `<button v-if="visible" class="stub-confirm" @click="$emit('confirm')">Confirm</button>`,
    }),
}))

// Stub TotpEnrollmentWizard to avoid complex wizard interactions in unrelated tests.
vi.mock('@/components/TotpEnrollmentWizard.vue', () => ({
    default: defineComponent({
        name: 'TotpEnrollmentWizard',
        emits: ['done'],
        template: `<div class="stub-totp-wizard"><button class="wizard-done-btn" @click="$emit('done')">Done</button></div>`,
    }),
}))

import {
    makeAuthMock as baseMakeAuthMock,
    makeDateFormatMock as baseDateFormatMock,
} from './helpers/mockUseAuth'
vi.mock('@/api/me')
vi.mock('@/api/totp')
vi.mock('@/api/db')

import AccountView from '@/views/AccountView.vue'
import DeleteConfirmDialog from '@/components/DeleteConfirmDialog.vue'
import { useAuth } from '@/composables/useAuth'
import { useDateFormat } from '@/composables/useDateFormat'
import { useRoute, useRouter } from 'vue-router'
import { getMe, changePassword, getPreferences, unlinkGoogle, deleteMyData } from '@/api/me'
import { getTotpStatus, disableTotp, regenerateRecoveryCodes } from '@/api/totp'
import { getResetJobStatus } from '@/api/db'
import { POLL_INTERVAL_MS } from '@/composables/useResetJob'

enableAutoUnmount(afterEach)

const ME_WITH_PASSWORD = {
    id: 1,
    email: 'user@example.com',
    display_name: 'Test User',
    has_password: true,
    google_linked: false,
    role: 'member',
    totp_enabled: false,
    totp_enrollment_required: false,
    recovery_codes_remaining: null,
}

const ME_WITHOUT_PASSWORD = {
    id: 1,
    email: 'google@example.com',
    display_name: null,
    has_password: false,
    google_linked: true,
    role: 'member',
    totp_enabled: false,
    totp_enrollment_required: false,
    recovery_codes_remaining: null,
}

const TOTP_STATUS_DISABLED = { enabled: false, enabled_at: null, recovery_codes_remaining: null }
const TOTP_STATUS_ENABLED = {
    enabled: true,
    enabled_at: '2026-01-01T00:00:00Z',
    recovery_codes_remaining: 10,
}
const TOTP_STATUS_LOW_CODES = {
    enabled: true,
    enabled_at: '2026-01-01T00:00:00Z',
    recovery_codes_remaining: 2,
}

const PREFS = { landing_page: 'dashboard' as const, date_format: 'iso' as const }

function makeAuthMock(role = 'member') {
    vi.mocked(useAuth).mockReturnValue(
        baseMakeAuthMock({
            role: ref(role) as never,
            isAuthenticated: ref(true) as never,
        }) as never,
    )
}

function makeDateFormatMock() {
    vi.mocked(useDateFormat).mockReturnValue(baseDateFormatMock() as never)
}

describe('AccountView — password form', () => {
    beforeEach(() => {
        vi.resetAllMocks()
        makeAuthMock()
        makeDateFormatMock()
        vi.mocked(useRoute).mockReturnValue({ query: {} } as never)
        vi.mocked(getMe).mockResolvedValue(ME_WITH_PASSWORD)
        vi.mocked(getPreferences).mockResolvedValue(PREFS as never)
        vi.mocked(getTotpStatus).mockResolvedValue(TOTP_STATUS_DISABLED)
    })

    async function mountAndLoad() {
        const wrapper = mount(AccountView)
        await flushPromises()
        return wrapper
    }

    it('test_password_mismatch_shows_error_and_does_not_call_changePassword', async () => {
        const wrapper = await mountAndLoad()

        await wrapper.find('#new-password').setValue('abc')
        await wrapper.find('#confirm-password').setValue('xyz')
        // The password stacked-form is the first .stacked-form in the template
        await wrapper.findAll('.stacked-form')[0].trigger('submit')
        await wrapper.vm.$nextTick()

        expect(wrapper.find('[role="alert"]').text()).toContain('Passwords do not match')
        expect(changePassword).not.toHaveBeenCalled()
    })

    it('test_empty_new_password_shows_error_and_does_not_call_changePassword', async () => {
        const wrapper = await mountAndLoad()

        // Leave #new-password at its default empty value; just submit
        await wrapper.findAll('.stacked-form')[0].trigger('submit')
        await wrapper.vm.$nextTick()

        expect(wrapper.find('[role="alert"]').text()).toContain('New password cannot be empty')
        expect(changePassword).not.toHaveBeenCalled()
    })

    it('test_successful_change_shows_success_clears_fields_and_calls_changePassword_with_current_password', async () => {
        vi.mocked(changePassword).mockResolvedValueOnce({ message: 'ok' })

        const wrapper = await mountAndLoad()
        await wrapper.find('#current-password').setValue('oldpass')
        await wrapper.find('#new-password').setValue('newpass123')
        await wrapper.find('#confirm-password').setValue('newpass123')
        await wrapper.findAll('.stacked-form')[0].trigger('submit')
        await flushPromises()

        expect(changePassword).toHaveBeenCalledWith({
            current_password: 'oldpass',
            new_password: 'newpass123',
        })
        expect(wrapper.find('.form-success').text()).toBe('Password updated')
        expect((wrapper.find('#new-password').element as HTMLInputElement).value).toBe('')
        expect((wrapper.find('#confirm-password').element as HTMLInputElement).value).toBe('')
    })

    it('test_google_only_account_hides_current_password_field_and_omits_it_from_changePassword', async () => {
        vi.mocked(getMe).mockResolvedValue(ME_WITHOUT_PASSWORD)
        vi.mocked(changePassword).mockResolvedValueOnce({ message: 'ok' })

        const wrapper = await mountAndLoad()

        expect(wrapper.find('#current-password').exists()).toBe(false)

        await wrapper.find('#new-password').setValue('newpass123')
        await wrapper.find('#confirm-password').setValue('newpass123')
        await wrapper.findAll('.stacked-form')[0].trigger('submit')
        await flushPromises()

        expect(changePassword).toHaveBeenCalledWith({ new_password: 'newpass123' })
        expect(changePassword).not.toHaveBeenCalledWith(
            expect.objectContaining({ current_password: expect.anything() }),
        )
    })

    it('test_changePassword_rejection_shows_error_and_suppresses_success', async () => {
        vi.mocked(changePassword).mockRejectedValueOnce(new Error('Unauthorized'))

        const wrapper = await mountAndLoad()
        await wrapper.find('#current-password').setValue('wrong')
        await wrapper.find('#new-password').setValue('newpass123')
        await wrapper.find('#confirm-password').setValue('newpass123')
        await wrapper.findAll('.stacked-form')[0].trigger('submit')
        await flushPromises()

        expect(wrapper.find('[role="alert"]').text()).toContain('Unauthorized')
        expect(wrapper.find('.form-success').exists()).toBe(false)
    })

    it('test_success_flips_has_password_locally_without_refetch', async () => {
        // A successful change must not depend on a follow-up network fetch:
        // has_password is updated locally, so a flaky connection can't hide
        // the success banner or blank the page into the meError state.
        vi.mocked(getMe).mockResolvedValue(ME_WITHOUT_PASSWORD)
        vi.mocked(changePassword).mockResolvedValueOnce({ message: 'ok' })

        const wrapper = await mountAndLoad()
        expect(wrapper.find('#current-password').exists()).toBe(false)

        await wrapper.find('#new-password').setValue('newpass123')
        await wrapper.find('#confirm-password').setValue('newpass123')
        await wrapper.findAll('.stacked-form')[0].trigger('submit')
        await flushPromises()

        // Success banner visible; no error rendered.
        expect(wrapper.find('.form-success').exists()).toBe(true)
        expect(wrapper.find('[role="alert"]').exists()).toBe(false)
        // getMe was called only for the initial mount — no refetch.
        expect(getMe).toHaveBeenCalledTimes(1)
        // has_password flipped locally: the current-password field now renders.
        expect(wrapper.find('#current-password').exists()).toBe(true)
    })

    it('test_demo_role_shows_banner_and_disables_password_inputs', async () => {
        makeAuthMock('demo')

        const wrapper = await mountAndLoad()

        expect(wrapper.find('.demo-banner').exists()).toBe(true)
        expect(wrapper.find('.demo-banner').attributes('role')).toBe('alert')
        expect((wrapper.find('#new-password').element as HTMLInputElement).disabled).toBe(true)
        expect((wrapper.find('#confirm-password').element as HTMLInputElement).disabled).toBe(true)
        expect((wrapper.find('#current-password').element as HTMLInputElement).disabled).toBe(true)
    })
})

describe('AccountView — sign-in methods / Google unlink', () => {
    const ME_LINKED_WITH_PW = {
        id: 1,
        email: 'user@example.com',
        display_name: 'Test User',
        has_password: true,
        google_linked: true,
        role: 'member',
        totp_enabled: false,
        totp_enrollment_required: false,
        recovery_codes_remaining: null,
    }
    const ME_LINKED_NO_PW = {
        id: 1,
        email: 'user@example.com',
        display_name: null,
        has_password: false,
        google_linked: true,
        role: 'member',
        totp_enabled: false,
        totp_enrollment_required: false,
        recovery_codes_remaining: null,
    }
    const ME_NOT_LINKED = {
        id: 1,
        email: 'user@example.com',
        display_name: 'Test User',
        has_password: true,
        google_linked: false,
        role: 'member',
        totp_enabled: false,
        totp_enrollment_required: false,
        recovery_codes_remaining: null,
    }

    let mockPush: ReturnType<typeof vi.fn>
    let mockReplace: ReturnType<typeof vi.fn>
    let mockClearAuth: ReturnType<typeof vi.fn>

    beforeEach(() => {
        vi.resetAllMocks()
        mockPush = vi.fn()
        mockReplace = vi.fn()
        mockClearAuth = vi.fn()
        vi.mocked(useRouter).mockReturnValue({ push: mockPush, replace: mockReplace } as never)
        vi.mocked(useRoute).mockReturnValue({ query: {} } as never)
        vi.mocked(useAuth).mockReturnValue(
            baseMakeAuthMock({
                role: ref('member') as never,
                isAuthenticated: ref(true) as never,
                clearAuth: mockClearAuth as never,
            }) as never,
        )
        vi.mocked(useDateFormat).mockReturnValue(baseDateFormatMock() as never)
        vi.mocked(getMe).mockResolvedValue(ME_LINKED_WITH_PW)
        vi.mocked(getPreferences).mockResolvedValue({
            landing_page: 'dashboard' as const,
            date_format: 'iso' as const,
        } as never)
        vi.mocked(getTotpStatus).mockResolvedValue(TOTP_STATUS_DISABLED)
    })

    async function mountAndLoad() {
        const wrapper = mount(AccountView)
        await flushPromises()
        return wrapper
    }

    it('test_google_linked_no_password_shows_warning_no_unlink_button', async () => {
        vi.mocked(getMe).mockResolvedValue(ME_LINKED_NO_PW)
        const wrapper = await mountAndLoad()

        // Inline warning must be visible.
        const alerts = wrapper.findAll('[role="alert"]')
        const warning = alerts.find((w) => w.text().includes('Set a password first'))
        expect(warning).toBeTruthy()

        // Unlink button must not be rendered when there is no password.
        const unlinkBtn = wrapper.findAll('button').find((b) => b.text().trim() === 'Unlink Google')
        expect(unlinkBtn).toBeUndefined()
    })

    it('test_google_linked_with_password_shows_unlink_button_and_opens_dialog', async () => {
        const wrapper = await mountAndLoad()

        // "Unlink Google" button present.
        const unlinkBtn = wrapper.findAll('button').find((b) => b.text().trim() === 'Unlink Google')
        expect(unlinkBtn).toBeTruthy()

        // Clicking it opens the confirm dialog.
        await unlinkBtn!.trigger('click')
        await wrapper.vm.$nextTick()

        const dialog = wrapper.findComponent(DeleteConfirmDialog)
        expect(dialog.props('visible')).toBe(true)
    })

    it('test_google_not_linked_shows_not_linked_no_button', async () => {
        vi.mocked(getMe).mockResolvedValue(ME_NOT_LINKED)
        const wrapper = await mountAndLoad()

        // Status text shows "Not linked".
        expect(wrapper.text()).toContain('Not linked')

        // No Unlink button rendered.
        const unlinkBtn = wrapper.findAll('button').find((b) => b.text().trim() === 'Unlink Google')
        expect(unlinkBtn).toBeUndefined()
    })

    it('test_confirm_unlink_success_calls_clearAuth_and_navigates', async () => {
        vi.mocked(unlinkGoogle).mockResolvedValueOnce({ message: 'Google account unlinked' })
        const wrapper = await mountAndLoad()

        // Open the dialog via the Unlink Google button.
        const unlinkBtn = wrapper.findAll('button').find((b) => b.text().trim() === 'Unlink Google')
        await unlinkBtn!.trigger('click')
        await wrapper.vm.$nextTick()

        // Emit confirm on the dialog to trigger confirmUnlinkGoogle.
        await wrapper.findComponent(DeleteConfirmDialog).vm.$emit('confirm')
        await flushPromises()

        expect(mockClearAuth).toHaveBeenCalledOnce()
        expect(mockPush).toHaveBeenCalledWith('/')
    })

    it('test_confirm_unlink_failure_shows_inline_error_no_navigation', async () => {
        vi.mocked(unlinkGoogle).mockRejectedValueOnce(new Error('Server error'))
        const wrapper = await mountAndLoad()

        // Open dialog and confirm.
        const unlinkBtn = wrapper.findAll('button').find((b) => b.text().trim() === 'Unlink Google')
        await unlinkBtn!.trigger('click')
        await wrapper.vm.$nextTick()
        await wrapper.findComponent(DeleteConfirmDialog).vm.$emit('confirm')
        await flushPromises()

        // Inline error shown; no navigation.
        const alerts = wrapper.findAll('[role="alert"]')
        const errorAlert = alerts.find((w) => w.text().includes('Server error'))
        expect(errorAlert).toBeTruthy()
        expect(mockClearAuth).not.toHaveBeenCalled()
        expect(mockPush).not.toHaveBeenCalled()
    })

    it('test_google_not_linked_shows_connect_google_link', async () => {
        vi.mocked(getMe).mockResolvedValue(ME_NOT_LINKED)
        const wrapper = await mountAndLoad()

        const links = wrapper.findAll('a')
        const connectLink = links.find((a) => a.text().includes('Connect Google'))
        expect(connectLink).toBeTruthy()
        expect(connectLink!.attributes('href')).toBe('/api/v1/auth/google/connect')

        const unlinkBtn = wrapper.findAll('button').find((b) => b.text().trim() === 'Unlink Google')
        expect(unlinkBtn).toBeUndefined()
    })

    it('test_google_not_linked_demo_shows_disabled_connect_button', async () => {
        vi.mocked(useAuth).mockReturnValue(
            baseMakeAuthMock({
                role: ref('demo') as never,
                isAuthenticated: ref(true) as never,
                clearAuth: mockClearAuth as never,
            }) as never,
        )
        vi.mocked(getMe).mockResolvedValue(ME_NOT_LINKED)
        const wrapper = await mountAndLoad()

        const connectBtn = wrapper
            .findAll('button')
            .find((b) => b.text().includes('Connect Google'))
        expect(connectBtn).toBeTruthy()
        expect((connectBtn!.element as HTMLButtonElement).disabled).toBe(true)

        const links = wrapper.findAll('a')
        const connectLink = links.find((a) => a.text().includes('Connect Google'))
        expect(connectLink).toBeUndefined()
    })

    it('test_google_connected_query_shows_success_and_cleans_url', async () => {
        vi.mocked(useRoute).mockReturnValue({ query: { google_connected: '1' } } as never)
        vi.mocked(getMe).mockResolvedValue(ME_LINKED_WITH_PW)
        const wrapper = await mountAndLoad()

        expect(wrapper.text()).toContain('Google account linked')
        expect(mockReplace).toHaveBeenCalledWith({ path: '/account' })
    })

    it('test_google_connect_error_query_shows_error_message', async () => {
        vi.mocked(useRoute).mockReturnValue({ query: { google_connect_error: '1' } } as never)
        vi.mocked(getMe).mockResolvedValue(ME_NOT_LINKED)
        const wrapper = await mountAndLoad()

        const alerts = wrapper.findAll('[role="alert"]')
        const errorAlert = alerts.find((a) => a.text().includes("Couldn't link Google"))
        expect(errorAlert).toBeTruthy()
        expect(mockReplace).toHaveBeenCalledWith({ path: '/account' })
    })
})

describe('AccountView — danger zone (delete-data)', () => {
    const DELETE_RESULT = {
        status: 'success',
        devices_deleted: 2,
        import_jobs_deleted: 5,
        profiles_processed: 1,
        size_before_mb: 10.0,
        size_after_mb: 1.5,
    }

    function jobStatus(overrides: Record<string, unknown> = {}) {
        return {
            kind: 'delete_data',
            state: 'running',
            error_message: null,
            reset_result: null,
            delete_data_result: null,
            ...overrides,
        } as never
    }

    beforeEach(() => {
        // flushPromises schedules through setImmediate, so that stays real.
        vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] })
        vi.resetAllMocks()
        sessionStorage.clear()
        makeAuthMock()
        makeDateFormatMock()
        vi.mocked(useRoute).mockReturnValue({ query: {} } as never)
        vi.mocked(getMe).mockResolvedValue(ME_WITH_PASSWORD)
        vi.mocked(getPreferences).mockResolvedValue(PREFS as never)
        vi.mocked(getTotpStatus).mockResolvedValue(TOTP_STATUS_DISABLED)
        vi.mocked(deleteMyData).mockResolvedValue({ job_id: 'job-token', state: 'queued' })
        vi.mocked(getResetJobStatus).mockResolvedValue(jobStatus())
    })

    afterEach(() => {
        vi.clearAllTimers()
        vi.useRealTimers()
        sessionStorage.clear()
    })

    async function mountAndLoad() {
        const wrapper = mount(AccountView)
        await flushPromises()
        return wrapper
    }

    async function confirmDeleteData(wrapper: Awaited<ReturnType<typeof mountAndLoad>>) {
        await wrapper.find('.danger-zone-card button').trigger('click')
        await wrapper.vm.$nextTick()
        await wrapper.find('.stub-confirm').trigger('click')
        await flushPromises()
    }

    async function pollOnce() {
        await vi.advanceTimersByTimeAsync(POLL_INTERVAL_MS)
        await flushPromises()
    }

    it('test_danger_zone_visible_for_non_demo', async () => {
        const wrapper = await mountAndLoad()
        expect(wrapper.find('.danger-zone-card').exists()).toBe(true)
    })

    it('test_danger_zone_hidden_for_demo', async () => {
        makeAuthMock('demo')
        const wrapper = await mountAndLoad()
        expect(wrapper.find('.danger-zone-card').exists()).toBe(false)
    })

    it('test_confirm_calls_deleteMyData_and_shows_success', async () => {
        const wrapper = await mountAndLoad()

        // The stub-confirm button only renders when deleteDataDialogOpen is true.
        expect(wrapper.find('.stub-confirm').exists()).toBe(false)

        // Open dialog — sets deleteDataDialogOpen = true.
        await wrapper.find('.danger-zone-card button').trigger('click')
        await wrapper.vm.$nextTick()

        expect(wrapper.find('.stub-confirm').exists()).toBe(true)

        // Fire confirm — triggers handleDeleteData.
        await wrapper.find('.stub-confirm').trigger('click')
        await flushPromises()

        expect(deleteMyData).toHaveBeenCalledTimes(1)
        expect(wrapper.find('.form-success').exists()).toBe(false)

        vi.mocked(getResetJobStatus).mockResolvedValue(
            jobStatus({ state: 'succeeded', delete_data_result: DELETE_RESULT }),
        )
        await pollOnce()

        expect(wrapper.find('.form-success').text()).toContain('2 device(s)')
        expect(wrapper.find('.form-success').text()).toContain('5 import record(s)')
        expect(wrapper.find('.form-success').text()).toContain('1 profile(s)')
        expect(wrapper.find('[role="status"]').exists()).toBe(false)
    })

    it('test_running_deletion_shows_notice_and_keeps_button_disabled', async () => {
        const wrapper = await mountAndLoad()
        await confirmDeleteData(wrapper)

        const button = wrapper.find('.danger-zone-card button')
        expect((button.element as HTMLButtonElement).disabled).toBe(true)
        expect(wrapper.find('[role="status"]').text()).toContain('several minutes')
    })

    it('test_deletion_resumes_after_returning_to_the_page', async () => {
        const firstVisit = await mountAndLoad()
        await confirmDeleteData(firstVisit)
        firstVisit.unmount()

        const wrapper = await mountAndLoad()

        expect(getResetJobStatus).toHaveBeenCalledWith('job-token')
        expect(wrapper.find('[role="status"]').exists()).toBe(true)
    })

    it('test_failed_deletion_shows_job_error_and_no_success', async () => {
        const wrapper = await mountAndLoad()
        await confirmDeleteData(wrapper)

        vi.mocked(getResetJobStatus).mockResolvedValue(
            jobStatus({ state: 'failed', error_message: 'The deletion failed.' }),
        )
        await pollOnce()

        expect(wrapper.find('[role="alert"]').text()).toContain('The deletion failed.')
        expect(wrapper.find('.form-success').exists()).toBe(false)
    })

    it('test_conflict_shows_in_progress_message', async () => {
        vi.mocked(deleteMyData).mockRejectedValueOnce(
            Object.assign(new Error('Conflict'), { response: { status: 409 } }),
        )

        const wrapper = await mountAndLoad()
        await confirmDeleteData(wrapper)

        expect(wrapper.find('[role="alert"]').text()).toContain(
            'Another reset or data deletion is in progress',
        )
    })

    it('test_deleteMyData_error_shows_inline_error', async () => {
        vi.mocked(deleteMyData).mockRejectedValueOnce(new Error('Server error'))

        const wrapper = await mountAndLoad()

        await wrapper.find('.danger-zone-card button').trigger('click')
        await wrapper.vm.$nextTick()
        await wrapper.find('.stub-confirm').trigger('click')
        await flushPromises()

        expect(wrapper.find('[role="alert"]').text()).toContain('Server error')
        expect(wrapper.find('.form-success').exists()).toBe(false)
    })
})

// ---------------------------------------------------------------------------
// TOTP card
// ---------------------------------------------------------------------------

describe('AccountView — two-factor authentication card', () => {
    let mockPush: ReturnType<typeof vi.fn>
    let mockClearAuth: ReturnType<typeof vi.fn>

    beforeEach(() => {
        vi.resetAllMocks()
        mockPush = vi.fn()
        mockClearAuth = vi.fn()
        vi.mocked(useRouter).mockReturnValue({ push: mockPush, replace: vi.fn() } as never)
        vi.mocked(useRoute).mockReturnValue({ query: {} } as never)
        vi.mocked(useAuth).mockReturnValue(
            baseMakeAuthMock({
                role: ref('member') as never,
                isAuthenticated: ref(true) as never,
                clearAuth: mockClearAuth as never,
            }) as never,
        )
        vi.mocked(useDateFormat).mockReturnValue(baseDateFormatMock() as never)
        vi.mocked(getMe).mockResolvedValue(ME_WITH_PASSWORD)
        vi.mocked(getPreferences).mockResolvedValue(PREFS as never)
        vi.mocked(getTotpStatus).mockResolvedValue(TOTP_STATUS_DISABLED)
    })

    async function mountAndLoad() {
        const wrapper = mount(AccountView)
        await flushPromises()
        return wrapper
    }

    it('test_2fa_card_visible_for_password_account', async () => {
        const wrapper = await mountAndLoad()
        expect(wrapper.text()).toContain('Two-factor authentication')
        expect(wrapper.text()).toContain('Set up two-factor auth')
    })

    it('test_2fa_card_hidden_for_google_only_account', async () => {
        vi.mocked(getMe).mockResolvedValue({
            ...ME_WITH_PASSWORD,
            has_password: false,
            google_linked: true,
            totp_enabled: false,
            totp_enrollment_required: false,
            recovery_codes_remaining: null,
        })
        const wrapper = await mountAndLoad()
        expect(wrapper.text()).not.toContain('Two-factor authentication')
    })

    it('test_2fa_card_hidden_for_demo_account', async () => {
        vi.mocked(useAuth).mockReturnValue(
            baseMakeAuthMock({
                role: ref('demo') as never,
                isAuthenticated: ref(true) as never,
            }) as never,
        )
        const wrapper = await mountAndLoad()
        expect(wrapper.text()).not.toContain('Two-factor authentication')
    })

    it('test_setup_button_shows_enrollment_wizard', async () => {
        const wrapper = await mountAndLoad()
        const setupBtn = wrapper
            .findAll('button')
            .find((b) => b.text().includes('Set up two-factor'))
        expect(setupBtn).toBeTruthy()
        await setupBtn!.trigger('click')
        await wrapper.vm.$nextTick()
        expect(wrapper.find('.stub-totp-wizard').exists()).toBe(true)
    })

    it('test_cancel_enrollment_wizard_returns_to_idle_state', async () => {
        const wrapper = await mountAndLoad()

        const setupBtn = wrapper
            .findAll('button')
            .find((b) => b.text().includes('Set up two-factor'))
        await setupBtn!.trigger('click')
        await wrapper.vm.$nextTick()

        expect(wrapper.find('.stub-totp-wizard').exists()).toBe(true)

        // Cancel button should be present when wizard is showing
        const cancelBtn = wrapper.findAll('button').find((b) => b.text().trim() === 'Cancel')
        expect(cancelBtn).toBeTruthy()

        await cancelBtn!.trigger('click')
        await wrapper.vm.$nextTick()

        // Wizard gone; idle state restored
        expect(wrapper.find('.stub-totp-wizard').exists()).toBe(false)
        expect(wrapper.text()).toContain('Set up two-factor auth')
    })

    it('test_wizard_done_reloads_totp_status_and_shows_enabled_badge', async () => {
        vi.mocked(getTotpStatus)
            .mockResolvedValueOnce(TOTP_STATUS_DISABLED)
            .mockResolvedValueOnce(TOTP_STATUS_ENABLED)

        const wrapper = await mountAndLoad()

        const setupBtn = wrapper
            .findAll('button')
            .find((b) => b.text().includes('Set up two-factor'))
        await setupBtn!.trigger('click')
        await wrapper.vm.$nextTick()

        // Fire done from stub wizard
        await wrapper.find('.wizard-done-btn').trigger('click')
        await flushPromises()

        expect(getTotpStatus).toHaveBeenCalledTimes(2)
        // Wizard hidden; enrolled state rendered
        expect(wrapper.find('.stub-totp-wizard').exists()).toBe(false)
        expect(wrapper.text()).toContain('Enabled')
    })

    it('test_enrolled_state_shows_enabled_badge_and_code_count', async () => {
        vi.mocked(getTotpStatus).mockResolvedValue(TOTP_STATUS_ENABLED)
        const wrapper = await mountAndLoad()
        expect(wrapper.text()).toContain('Enabled')
        expect(wrapper.text()).toContain('10 remaining')
    })

    it('test_low_codes_warning_shown_when_3_or_fewer_remain', async () => {
        vi.mocked(getTotpStatus).mockResolvedValue(TOTP_STATUS_LOW_CODES)
        const wrapper = await mountAndLoad()
        expect(wrapper.text()).toContain('2 remaining')
        expect(wrapper.text()).toContain('Few recovery codes left')
    })

    it('test_disable_success_calls_clearAuth_and_navigates_to_login', async () => {
        vi.mocked(getTotpStatus).mockResolvedValue(TOTP_STATUS_ENABLED)
        vi.mocked(disableTotp).mockResolvedValueOnce({ message: 'TOTP disabled' })

        const wrapper = await mountAndLoad()

        // Open the disable form
        const disableBtn = wrapper
            .findAll('button')
            .find((b) => b.text().includes('Disable two-factor'))
        expect(disableBtn).toBeTruthy()
        await disableBtn!.trigger('click')
        await wrapper.vm.$nextTick()

        // Fill in password and code
        await wrapper.find('#disable-password').setValue('mypassword')
        await wrapper.find('#disable-code').setValue('123456')

        // Trigger submit on the stacked-form that wraps the disable fields
        const forms = wrapper.findAll('.stacked-form')
        const disableStacked = forms.find((f) => f.find('#disable-code').exists())
        expect(disableStacked).toBeTruthy()
        await disableStacked!.trigger('submit')
        await flushPromises()

        expect(disableTotp).toHaveBeenCalledWith({ password: 'mypassword', code: '123456' })
        expect(mockClearAuth).toHaveBeenCalledOnce()
        expect(mockPush).toHaveBeenCalledWith('/')
    })

    it('test_disable_code_is_normalized_before_submit', async () => {
        vi.mocked(getTotpStatus).mockResolvedValue(TOTP_STATUS_ENABLED)
        vi.mocked(disableTotp).mockResolvedValueOnce({ message: 'TOTP disabled' })

        const wrapper = await mountAndLoad()

        const disableBtn = wrapper
            .findAll('button')
            .find((b) => b.text().includes('Disable two-factor'))
        await disableBtn!.trigger('click')
        await wrapper.vm.$nextTick()

        // Enter a recovery code with mixed case and whitespace
        await wrapper.find('#disable-password').setValue('mypassword')
        await wrapper.find('#disable-code').setValue('  ABCDEF01  ')

        const forms = wrapper.findAll('.stacked-form')
        const disableStacked = forms.find((f) => f.find('#disable-code').exists())
        await disableStacked!.trigger('submit')
        await flushPromises()

        // Code must be trimmed and lowercased
        expect(disableTotp).toHaveBeenCalledWith({ password: 'mypassword', code: 'abcdef01' })
    })

    it('test_disable_failure_shows_inline_error_no_navigation', async () => {
        vi.mocked(getTotpStatus).mockResolvedValue(TOTP_STATUS_ENABLED)
        vi.mocked(disableTotp).mockRejectedValueOnce(new Error('Wrong code'))

        const wrapper = await mountAndLoad()

        const disableBtn = wrapper
            .findAll('button')
            .find((b) => b.text().includes('Disable two-factor'))
        await disableBtn!.trigger('click')
        await wrapper.vm.$nextTick()

        await wrapper.find('#disable-password').setValue('pw')
        await wrapper.find('#disable-code').setValue('000000')
        const forms = wrapper.findAll('.stacked-form')
        const disableStacked = forms.find((f) => f.find('#disable-code').exists())
        await disableStacked!.trigger('submit')
        await flushPromises()

        const alerts = wrapper.findAll('[role="alert"]')
        const errorAlert = alerts.find((a) => a.text().includes('Wrong code'))
        expect(errorAlert).toBeTruthy()
        expect(mockClearAuth).not.toHaveBeenCalled()
        expect(mockPush).not.toHaveBeenCalled()
    })

    it('test_regen_success_shows_new_codes', async () => {
        vi.mocked(getTotpStatus).mockResolvedValue(TOTP_STATUS_ENABLED)
        vi.mocked(regenerateRecoveryCodes).mockResolvedValueOnce({
            recovery_codes: ['code-a', 'code-b', 'code-c'],
        })

        const wrapper = await mountAndLoad()

        // Open regen form
        const regenBtn = wrapper
            .findAll('button')
            .find((b) => b.text().includes('Regenerate recovery'))
        expect(regenBtn).toBeTruthy()
        await regenBtn!.trigger('click')
        await wrapper.vm.$nextTick()

        // Find the regen stacked-form (the one without #disable-password)
        const forms = wrapper.findAll('.stacked-form')
        const regenStacked = forms.find(
            (f) =>
                !f.find('#disable-password').exists() &&
                f.find('input[inputmode="numeric"]').exists(),
        )
        expect(regenStacked).toBeTruthy()
        await regenStacked!.find('input[inputmode="numeric"]').setValue('654321')
        await regenStacked!.trigger('submit')
        await flushPromises()

        expect(regenerateRecoveryCodes).toHaveBeenCalledWith({ code: '654321' })
        expect(wrapper.text()).toContain('code-a')
        expect(wrapper.text()).toContain('code-b')
    })

    it('test_regen_done_disabled_until_acknowledged_and_cancel_clears_state', async () => {
        vi.mocked(getTotpStatus).mockResolvedValue(TOTP_STATUS_ENABLED)
        vi.mocked(regenerateRecoveryCodes).mockResolvedValueOnce({
            recovery_codes: ['code-x', 'code-y'],
        })

        const wrapper = await mountAndLoad()

        // Open regen and submit
        const regenBtn = wrapper
            .findAll('button')
            .find((b) => b.text().includes('Regenerate recovery'))
        await regenBtn!.trigger('click')
        await wrapper.vm.$nextTick()

        const forms = wrapper.findAll('.stacked-form')
        const regenStacked = forms.find(
            (f) =>
                !f.find('#disable-password').exists() &&
                f.find('input[inputmode="numeric"]').exists(),
        )
        await regenStacked!.find('input[inputmode="numeric"]').setValue('654321')
        await regenStacked!.trigger('submit')
        await flushPromises()

        expect(wrapper.text()).toContain('code-x')

        // Done button disabled (no acknowledge yet)
        const doneBtn = wrapper.findAll('button').find((b) => b.text().trim() === 'Done')
        expect(doneBtn).toBeTruthy()
        expect((doneBtn!.element as HTMLButtonElement).disabled).toBe(true)

        // Check the acknowledge checkbox (rendered inside RecoveryCodesDisplay)
        await wrapper.find('input[type="checkbox"]').setValue(true)
        await wrapper.vm.$nextTick()

        expect((doneBtn!.element as HTMLButtonElement).disabled).toBe(false)

        // Click Cancel
        const cancelBtn = wrapper.findAll('button').find((b) => b.text().trim() === 'Cancel')
        expect(cancelBtn).toBeTruthy()
        await cancelBtn!.trigger('click')
        await wrapper.vm.$nextTick()

        // State cleared: codes gone, regen form gone
        expect(wrapper.text()).not.toContain('code-x')
        expect(wrapper.find('input[inputmode="numeric"]').exists()).toBe(false)
    })

    it('test_regen_api_rejection_shows_inline_error_and_keeps_form_visible', async () => {
        vi.mocked(getTotpStatus).mockResolvedValue(TOTP_STATUS_ENABLED)
        vi.mocked(regenerateRecoveryCodes).mockRejectedValueOnce(new Error('Invalid code'))

        const wrapper = await mountAndLoad()

        const regenBtn = wrapper
            .findAll('button')
            .find((b) => b.text().includes('Regenerate recovery'))
        await regenBtn!.trigger('click')
        await wrapper.vm.$nextTick()

        const forms = wrapper.findAll('.stacked-form')
        const regenStacked = forms.find(
            (f) =>
                !f.find('#disable-password').exists() &&
                f.find('input[inputmode="numeric"]').exists(),
        )
        await regenStacked!.find('input[inputmode="numeric"]').setValue('654321')
        await regenStacked!.trigger('submit')
        await flushPromises()

        // Inline error visible
        const alerts = wrapper.findAll('[role="alert"]')
        const errAlert = alerts.find((a) => a.text().includes('Invalid code'))
        expect(errAlert).toBeTruthy()

        // Regen form still visible
        expect(wrapper.find('input[inputmode="numeric"]').exists()).toBe(true)
    })

    it('test_regen_copy_all_failure_shows_fallback_message', async () => {
        vi.mocked(getTotpStatus).mockResolvedValue(TOTP_STATUS_ENABLED)
        vi.mocked(regenerateRecoveryCodes).mockResolvedValueOnce({
            recovery_codes: ['code-a'],
        })

        const wrapper = await mountAndLoad()

        const regenBtn = wrapper
            .findAll('button')
            .find((b) => b.text().includes('Regenerate recovery'))
        await regenBtn!.trigger('click')
        await wrapper.vm.$nextTick()

        const forms = wrapper.findAll('.stacked-form')
        const regenStacked = forms.find(
            (f) =>
                !f.find('#disable-password').exists() &&
                f.find('input[inputmode="numeric"]').exists(),
        )
        await regenStacked!.find('input[inputmode="numeric"]').setValue('654321')
        await regenStacked!.trigger('submit')
        await flushPromises()

        // Mock clipboard to fail
        Object.defineProperty(navigator, 'clipboard', {
            value: { writeText: vi.fn().mockRejectedValueOnce(new Error('Not allowed')) },
            writable: true,
            configurable: true,
        })

        const copyBtn = wrapper.findAll('button').find((b) => b.text().includes('Copy all'))
        expect(copyBtn).toBeTruthy()
        await copyBtn!.trigger('click')
        await flushPromises()

        const alerts = wrapper.findAll('[role="alert"]')
        const fallbackAlert = alerts.find((a) => a.text().includes('Copy failed'))
        expect(fallbackAlert).toBeTruthy()
    })
})
