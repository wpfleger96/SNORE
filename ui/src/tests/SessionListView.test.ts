import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'

vi.mock('vue-router')
vi.mock('@/composables/useAuth')

vi.mock('@/api/sessions', () => ({
    getSessions: vi.fn(),
    updateSession: vi.fn(),
    deleteSessions: vi.fn(),
    getSessionDeletePreview: vi.fn(),
    getBulkDeletePreview: vi.fn(),
}))

vi.mock('@/api/devices', () => ({ getDevices: vi.fn() }))

vi.mock('@/composables/useAvailableDates', () => ({
    useAvailableDates: () => ({
        load: vi.fn(),
        isDateDisabled: () => false,
        minValue: undefined,
        maxValue: undefined,
    }),
}))

vi.mock('@/components/DatePickerInput.vue', () => ({
    default: { props: ['modelValue'], template: '<input :value="modelValue" />' },
}))

import { makeAuthMock } from './helpers/mockUseAuth'
import { chooseSelectOption, openSelect, selectedOptionLabel } from './helpers/selectOption'
import { useAuth } from '@/composables/useAuth'
import { useRoute, useRouter } from 'vue-router'
import { getSessions } from '@/api/sessions'
import { getDevices } from '@/api/devices'
import SessionListView from '@/views/SessionListView.vue'

const DEVICE = { manufacturer: 'ResMed', model: 'AirSense 11', serial_number: 'SN1' }

describe('SessionListView device filter', () => {
    let wrapper: VueWrapper

    beforeEach(() => {
        vi.mocked(useAuth).mockReturnValue(makeAuthMock() as never)
        vi.mocked(useRoute).mockReturnValue({ query: {} } as never)
        vi.mocked(useRouter).mockReturnValue({ push: vi.fn() } as never)
        vi.mocked(getSessions).mockResolvedValue({ items: [], total: 0 } as never)
        vi.mocked(getDevices).mockResolvedValue([DEVICE] as never)
    })

    afterEach(() => {
        wrapper?.unmount()
        vi.clearAllMocks()
    })

    function mountView(): VueWrapper {
        return mount(SessionListView, {
            attachTo: document.body,
            global: { stubs: { RouterLink: true } },
        })
    }

    it('test_device_filter_starts_on_all_devices', async () => {
        wrapper = mountView()
        await flushPromises()

        expect(vi.mocked(getSessions).mock.calls.at(-1)![0]?.device).toBeUndefined()
        expect(selectedOptionLabel(await openSelect(wrapper))).toBe('All devices')
    })

    it('test_choosing_all_devices_after_a_device_sends_no_device_param', async () => {
        wrapper = mountView()
        await flushPromises()

        await chooseSelectOption(wrapper, 'ResMed AirSense 11')
        expect(vi.mocked(getSessions).mock.calls.at(-1)![0]?.device).toBe('ResMed AirSense 11')

        await chooseSelectOption(wrapper, 'All devices')
        expect(vi.mocked(getSessions).mock.calls.at(-1)![0]?.device).toBeUndefined()
    })
})
