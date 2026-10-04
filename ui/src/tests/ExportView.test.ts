import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'

vi.mock('@/api/devices', () => ({ getDevices: vi.fn() }))

vi.mock('@/api/export', () => ({
    exportCsv: vi.fn(),
    exportJson: vi.fn(),
    exportRaw: vi.fn(),
    downloadBlob: vi.fn(),
}))

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

import { chooseSelectOption, openSelect, selectedOptionLabel } from './helpers/selectOption'
import { getDevices } from '@/api/devices'
import { exportCsv } from '@/api/export'
import ExportView from '@/views/ExportView.vue'

const DEVICE = { manufacturer: 'ResMed', model: 'AirSense 11', serial_number: 'SN1' }

async function clickExport(wrapper: VueWrapper): Promise<void> {
    const button = wrapper.findAll('button').find((b) => b.text().trim() === 'Export')
    await button!.trigger('click')
    await flushPromises()
}

describe('ExportView device filter', () => {
    let wrapper: VueWrapper

    beforeEach(() => {
        vi.mocked(getDevices).mockResolvedValue([DEVICE] as never)
        vi.mocked(exportCsv).mockResolvedValue(new Blob(['x']))
    })

    afterEach(() => {
        wrapper?.unmount()
        vi.clearAllMocks()
    })

    it('test_device_filter_starts_on_all_devices', async () => {
        wrapper = mount(ExportView, { attachTo: document.body })
        await flushPromises()

        expect(selectedOptionLabel(await openSelect(wrapper))).toBe('All Devices')
    })

    it('test_choosing_all_devices_after_a_device_sends_no_device_param', async () => {
        wrapper = mount(ExportView, { attachTo: document.body })
        await flushPromises()

        await chooseSelectOption(wrapper, 'ResMed AirSense 11')
        await clickExport(wrapper)
        expect(vi.mocked(exportCsv).mock.calls.at(-1)![0]?.device).toBe('ResMed AirSense 11')

        await chooseSelectOption(wrapper, 'All Devices')
        await clickExport(wrapper)
        expect(vi.mocked(exportCsv).mock.calls.at(-1)![0]).not.toHaveProperty('device')
    })
})
