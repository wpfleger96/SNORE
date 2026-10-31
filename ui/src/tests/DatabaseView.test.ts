import { describe, expect, it, vi, beforeEach } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
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
import { getDbStats, resetDb } from '@/api/db'

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

const RESET_RESULT_DATA_ONLY = {
    status: 'success',
    tables_cleared: { sessions: 30, devices: 1 },
    total_rows_deleted: 31,
    size_before_mb: 42.5,
    size_after_mb: null,
    vacuum_scheduled: true,
    bootstrap_invite_url: null,
}

const RESET_RESULT_FULL = {
    status: 'success',
    tables_cleared: { users: 1, sessions: 30, devices: 1 },
    total_rows_deleted: 32,
    size_before_mb: 42.5,
    size_after_mb: null,
    vacuum_scheduled: true,
    bootstrap_invite_url: '/invite#abc123',
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
        vi.resetAllMocks()
        makeAuthMock()
        vi.mocked(getDbStats).mockResolvedValue(DB_STATS as never)
        vi.mocked(resetDb).mockResolvedValue(RESET_RESULT_DATA_ONLY as never)
    })

    async function mountAndLoad() {
        const wrapper = mount(DatabaseView)
        await flushPromises()
        return wrapper
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

    it('test_data_only_reset_does_not_show_invite_banner', async () => {
        const wrapper = await mountAndLoad()

        await wrapper.find('.danger-card button').trigger('click')
        await wrapper.vm.$nextTick()
        await wrapper.find('.stub-confirm').trigger('click')
        await flushPromises()

        expect(wrapper.find('.invite-banner').exists()).toBe(false)
    })

    it('test_full_reset_shows_invite_banner_and_does_not_reload', async () => {
        vi.mocked(resetDb).mockResolvedValueOnce(RESET_RESULT_FULL as never)

        const wrapper = await mountAndLoad()

        // Check the include_accounts checkbox.
        await wrapper.find('.include-accounts-checkbox').setValue(true)
        await wrapper.vm.$nextTick()

        await wrapper.find('.danger-card button').trigger('click')
        await wrapper.vm.$nextTick()
        await wrapper.find('.stub-confirm').trigger('click')
        await flushPromises()

        // Invite banner must be visible with the URL.
        expect(wrapper.find('.invite-banner').exists()).toBe(true)
        expect(wrapper.find('.invite-url-text').text()).toContain('/invite#abc123')

        // getDbStats must only have been called once (initial load) — no
        // reload after full reset because the dead session would redirect away.
        expect(getDbStats).toHaveBeenCalledTimes(1)
    })

    it('test_full_reset_does_not_set_error_when_invite_url_present', async () => {
        vi.mocked(resetDb).mockResolvedValueOnce(RESET_RESULT_FULL as never)

        const wrapper = await mountAndLoad()

        await wrapper.find('.include-accounts-checkbox').setValue(true)
        await wrapper.vm.$nextTick()
        await wrapper.find('.danger-card button').trigger('click')
        await wrapper.vm.$nextTick()
        await wrapper.find('.stub-confirm').trigger('click')
        await flushPromises()

        // Invite banner visible; no error shown.
        expect(wrapper.find('.invite-banner').exists()).toBe(true)
        expect(wrapper.find('.error-state').exists()).toBe(false)
    })

    it('test_reset_error_shows_when_no_invite_url', async () => {
        vi.mocked(resetDb).mockRejectedValueOnce(new Error('Server error'))

        const wrapper = await mountAndLoad()

        await wrapper.find('.danger-card button').trigger('click')
        await wrapper.vm.$nextTick()
        await wrapper.find('.stub-confirm').trigger('click')
        await flushPromises()

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
