import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { enableAutoUnmount, flushPromises, mount } from '@vue/test-utils'

vi.mock('@/api/import', () => ({ triggerRescan: vi.fn() }))

import ArchiveRescanCard from '@/components/ArchiveRescanCard.vue'
import DatePickerInput from '@/components/DatePickerInput.vue'
import { triggerRescan } from '@/api/import'

enableAutoUnmount(afterEach)

const DEFAULT_BODY = {
    profile_id: 1,
    force: false,
    from_date: null,
    to_date: null,
    analyze: true,
}

function mountCard(profileId: number | null = 1) {
    // The confirmation dialog teleports to document.body.
    return mount(ArchiveRescanCard, { props: { profileId }, attachTo: document.body })
}

function rescanButton(wrapper: ReturnType<typeof mountCard>) {
    return wrapper.findAll('button').find((b) => b.text() === 'Re-import from archive')!
}

function bodyButton(label: string): HTMLButtonElement | undefined {
    return [...document.body.querySelectorAll('button')].find(
        (b) => b.textContent?.trim() === label,
    )
}

async function setDates(wrapper: ReturnType<typeof mountCard>, from: string, to: string) {
    const [fromPicker, toPicker] = wrapper.findAllComponents(DatePickerInput)
    fromPicker.vm.$emit('update:modelValue', from)
    toPicker.vm.$emit('update:modelValue', to)
    await flushPromises()
}

describe('ArchiveRescanCard', () => {
    beforeEach(() => {
        vi.resetAllMocks()
        vi.mocked(triggerRescan).mockResolvedValue({ job_id: 'rescan-1' })
    })

    it('test_plain_rescan_starts_at_once_with_defaults', async () => {
        const wrapper = mountCard()

        await rescanButton(wrapper).trigger('click')
        await flushPromises()

        expect(triggerRescan).toHaveBeenCalledExactlyOnceWith(DEFAULT_BODY)
        expect(wrapper.emitted('started')).toHaveLength(1)
    })

    it('test_sends_null_profile_when_none_selected', async () => {
        const wrapper = mountCard(null)

        await rescanButton(wrapper).trigger('click')
        await flushPromises()

        expect(triggerRescan).toHaveBeenCalledExactlyOnceWith({ ...DEFAULT_BODY, profile_id: null })
    })

    it('test_date_hint_explains_sessions_after_midnight', () => {
        const wrapper = mountCard()

        const hint = wrapper.find('.archive-hint').text()
        expect(hint).toContain('Dates select nights')
        expect(hint).toContain("belongs to the previous day's night")
    })

    it('test_sends_date_range_and_analysis_choice', async () => {
        const wrapper = mountCard()
        await setDates(wrapper, '2024-06-01', '2024-06-30')
        await wrapper.find('[data-testid="analyze"]').setValue(false)

        await rescanButton(wrapper).trigger('click')
        await flushPromises()

        expect(triggerRescan).toHaveBeenCalledExactlyOnceWith({
            ...DEFAULT_BODY,
            from_date: '2024-06-01',
            to_date: '2024-06-30',
            analyze: false,
        })
    })

    it('test_reversed_date_range_blocks_rescan', async () => {
        const wrapper = mountCard()
        await setDates(wrapper, '2024-07-01', '2024-06-30')

        expect(wrapper.text()).toContain('The From date must be on or before the To date.')
        expect(rescanButton(wrapper).attributes('disabled')).toBeDefined()
    })

    it('test_force_shows_warning', async () => {
        const wrapper = mountCard()
        expect(wrapper.find('[role="alert"]').exists()).toBe(false)

        await wrapper.find('[data-testid="force"]').setValue(true)

        const warning = wrapper.find('[role="alert"]').text()
        expect(warning).toContain('deleted and recreated')
        expect(warning).toContain('analysis results and breath rows')
        expect(warning).toContain('Disabled sessions stay disabled')
    })

    it('test_forced_rescan_waits_for_confirmation', async () => {
        const wrapper = mountCard()
        await wrapper.find('[data-testid="force"]').setValue(true)

        await rescanButton(wrapper).trigger('click')
        await flushPromises()

        expect(triggerRescan).not.toHaveBeenCalled()
        expect(document.body.textContent).toContain('Force re-parse?')
        expect(document.body.textContent).toContain('rebuilt from the files in the server archive')
        expect(bodyButton('Delete and re-parse')).toBeDefined()
    })

    it('test_forced_rescan_confirmation_repeats_date_hint_for_a_range', async () => {
        const wrapper = mountCard()
        await wrapper.find('[data-testid="force"]').setValue(true)
        await setDates(wrapper, '2024-06-20', '2024-06-20')

        await rescanButton(wrapper).trigger('click')
        await flushPromises()

        const dialog = document.body.querySelector('[role="alertdialog"]')!.textContent
        expect(dialog).toContain('Dates select nights')
    })

    it('test_confirmed_forced_rescan_sends_force_then_disarms_it', async () => {
        const wrapper = mountCard()
        await wrapper.find('[data-testid="force"]').setValue(true)
        await rescanButton(wrapper).trigger('click')
        await flushPromises()

        bodyButton('Delete and re-parse')!.click()
        await flushPromises()

        expect(triggerRescan).toHaveBeenCalledExactlyOnceWith({ ...DEFAULT_BODY, force: true })
        expect(wrapper.emitted('started')).toHaveLength(1)
        expect((wrapper.find('[data-testid="force"]').element as HTMLInputElement).checked).toBe(
            false,
        )
    })

    it('test_cancelled_confirmation_sends_nothing', async () => {
        const wrapper = mountCard()
        await wrapper.find('[data-testid="force"]').setValue(true)
        await rescanButton(wrapper).trigger('click')
        await flushPromises()

        bodyButton('Cancel')!.click()
        await flushPromises()

        expect(triggerRescan).not.toHaveBeenCalled()
        expect(bodyButton('Delete and re-parse')).toBeUndefined()
    })

    it('test_error_message_is_shown_and_nothing_emitted', async () => {
        // The API client's interceptor copies the server's `detail` into `message`.
        vi.mocked(triggerRescan).mockRejectedValue(
            new Error('Archive exists but contains no device data.'),
        )
        const wrapper = mountCard()

        await rescanButton(wrapper).trigger('click')
        await flushPromises()

        expect(wrapper.find('.error-text').text()).toBe(
            'Archive exists but contains no device data.',
        )
        expect(wrapper.emitted('started')).toBeUndefined()
    })

    it('test_double_click_starts_one_rescan', async () => {
        vi.mocked(triggerRescan).mockReturnValue(new Promise(() => {}))
        const wrapper = mountCard()

        void rescanButton(wrapper).trigger('click')
        void rescanButton(wrapper).trigger('click')
        await flushPromises()

        expect(triggerRescan).toHaveBeenCalledTimes(1)
    })
})
