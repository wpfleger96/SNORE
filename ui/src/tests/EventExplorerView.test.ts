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
    getEventMatch: vi.fn().mockResolvedValue({
        machine_count: 1,
        programmatic_count: 2,
        matched: 1,
        false_positives: 1,
        false_negatives: 0,
    }),
}))
vi.mock('@/api/sessions', () => ({
    getSession: vi.fn().mockResolvedValue({ duration_hours: 2 }),
}))

import EventExplorerView from '@/views/EventExplorerView.vue'

describe('EventExplorerView', () => {
    it('test_match_section_is_flagged_experimental_and_event_list_is_not', async () => {
        const wrapper = mount(EventExplorerView, {
            props: { sessionId: 3 },
            global: { stubs: { RouterLink: { template: '<a><slot /></a>' } } },
        })
        await flushPromises()

        const sections = wrapper.findAll('.section-card')
        const heading = (s: (typeof sections)[number]) => s.find('h2').text()
        expect(sections.map(heading)).toEqual([
            'Device-scored vs SNORE-detected',
            'Device-scored Events',
        ])
        expect(sections[0]!.find('[role="note"]').exists()).toBe(true)
        expect(sections[1]!.find('[role="note"]').exists()).toBe(false)
    })
})
