import { afterEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { setMediaMatches } from './matchMedia'

vi.mock('vue-router', () => ({ useRouter: () => ({ push: vi.fn() }) }))
vi.mock('@/api/events', () => ({
    getSessionEvents: vi.fn().mockResolvedValue([
        {
            id: 1,
            event_type: 'OA',
            start_time: 0,
            duration_seconds: 12,
            offset_seconds: 60,
            spo2_drop: null,
            peak_flow_limitation: null,
        },
        {
            id: 2,
            event_type: 'H',
            start_time: 0,
            duration_seconds: null,
            offset_seconds: 120,
            spo2_drop: null,
            peak_flow_limitation: null,
        },
    ]),
}))
vi.mock('@/api/sessions', () => ({
    getSession: vi.fn().mockResolvedValue({ duration_hours: 2 }),
}))

import EventExplorerView from '@/views/EventExplorerView.vue'

async function mountView() {
    const wrapper = mount(EventExplorerView, {
        props: { sessionId: 3 },
        global: { stubs: { RouterLink: { template: '<a><slot /></a>' } } },
    })
    await flushPromises()
    return wrapper
}

describe('EventExplorerView', () => {
    afterEach(() => setMediaMatches(false))

    it('test_renders_device_events_without_experimental_match_section', async () => {
        const wrapper = await mountView()

        const headings = wrapper.findAll('.section-card h2').map((h) => h.text())
        expect(headings).toEqual(['Device-scored Events'])
        expect(wrapper.find('[role="note"]').exists()).toBe(false)
        expect(wrapper.text()).toContain('OA')
        expect(wrapper.text()).toContain('Total Events')
    })

    it('test_desktop_table_shows_null_duration_as_em_dash', async () => {
        const wrapper = await mountView()

        const durations = wrapper.findAll('tbody tr').map((row) => row.findAll('td')[2].text())
        expect(durations).toEqual(['12.0s', '\u2014'])
    })

    it('test_mobile_cards_show_null_duration_as_em_dash', async () => {
        setMediaMatches(true)
        const wrapper = await mountView()

        const durations = wrapper.findAll('.data-card-value').map((v) => v.text())
        expect(durations).toEqual(['12.0s', '\u2014'])
    })

    it('test_desktop_table_drops_spo2_drop_and_peak_fl_columns', async () => {
        const wrapper = await mountView()

        expect(wrapper.findAll('th').map((th) => th.text())).toEqual(['Type', 'Time', 'Duration'])
    })

    it('test_mobile_cards_drop_spo2_drop_and_peak_fl_rows', async () => {
        setMediaMatches(true)
        const wrapper = await mountView()

        const labels = wrapper.findAll('.data-card-label').map((l) => l.text())
        expect(new Set(labels)).toEqual(new Set(['Duration']))
    })
})
