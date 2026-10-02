import { describe, expect, it, vi, beforeEach } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import type { DeviceDetail, DeviceInfo } from '@/types'

vi.mock('@/api/devices', () => ({
    getDevices: vi.fn(),
    getDeviceDetail: vi.fn(),
}))
vi.mock('@/api/equipment', () => ({ getMaskEpochs: vi.fn() }))
vi.mock('@/components/MaskLogManager.vue', () => ({
    default: { template: '<div />', props: ['epochs'] },
}))
// Stub the collapsible so its content renders eagerly (reka-ui otherwise gates
// slot content behind open state / presence transitions).
vi.mock('@/components/ui/collapsible', () => ({
    Collapsible: { template: '<div><slot /></div>' },
    CollapsibleTrigger: { template: '<div><slot /></div>' },
    CollapsibleContent: { template: '<div><slot /></div>' },
}))

import EquipmentView from '@/views/EquipmentView.vue'
import { getDevices, getDeviceDetail } from '@/api/devices'
import { getMaskEpochs } from '@/api/equipment'

function makeDevice(currentSettings: DeviceDetail['current_settings']): DeviceDetail {
    return {
        id: 1,
        manufacturer: 'ResMed',
        model: 'AirSense 11',
        serial_number: 'SN123',
        first_seen: '2026-01-01T00:00:00Z',
        usage: {
            session_count: 5,
            first_session_date: '2026-01-01',
            last_session_date: '2026-01-05',
            total_therapy_hours: 40,
            therapy_modes: ['APAP'],
        },
        current_settings: currentSettings,
        settings_history: [],
    }
}

async function mountWith(currentSettings: DeviceDetail['current_settings']) {
    vi.mocked(getDevices).mockResolvedValue([{ id: 1 } as DeviceInfo])
    vi.mocked(getDeviceDetail).mockResolvedValue(makeDevice(currentSettings))
    vi.mocked(getMaskEpochs).mockResolvedValue([])
    const wrapper = mount(EquipmentView)
    await flushPromises()
    return wrapper
}

describe('EquipmentView current settings', () => {
    beforeEach(() => {
        vi.resetAllMocks()
    })

    it('renders an InfoHint trigger for a known setting', async () => {
        const wrapper = await mountWith({ ipap: '14', some_unknown_key: 'x' })

        const infoButtons = wrapper
            .findAll('button')
            .filter((b) => b.attributes('aria-label')?.includes('IPAP'))

        expect(infoButtons).toHaveLength(1)
    })

    it('renders no InfoHint for an unknown setting row', async () => {
        const wrapper = await mountWith({ ipap: '14', some_unknown_key: 'x' })

        const unknownRow = wrapper
            .findAll('.setting-row')
            .find((row) => row.text().includes('some_unknown_key'))

        expect(unknownRow, 'unknown setting row should render').toBeDefined()
        expect(unknownRow!.find('button').exists()).toBe(false)
    })

    it('shows the empty-state message when current_settings is an empty object', async () => {
        const wrapper = await mountWith({})

        expect(wrapper.text()).toContain('No settings recorded for this device.')
    })
})
