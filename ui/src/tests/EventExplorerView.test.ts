import { describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'

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
    ]),
}))
vi.mock('@/api/sessions', () => ({
    getSession: vi.fn().mockResolvedValue({ duration_hours: 2 }),
}))

import EventExplorerView from '@/views/EventExplorerView.vue'

describe('EventExplorerView', () => {
    it('test_renders_device_events_without_experimental_match_section', async () => {
        const wrapper = mount(EventExplorerView, {
            props: { sessionId: 3 },
            global: { stubs: { RouterLink: { template: '<a><slot /></a>' } } },
        })
        await flushPromises()

        const headings = wrapper.findAll('.section-card h2').map((h) => h.text())
        expect(headings).toEqual(['Device-scored Events'])
        expect(wrapper.find('[role="note"]').exists()).toBe(false)
        expect(wrapper.text()).toContain('OA')
        expect(wrapper.text()).toContain('Total Events')
    })
})
