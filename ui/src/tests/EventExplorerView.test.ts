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

describe('EventExplorerView provenance', () => {
    it('test_event_list_is_headed_device_scored_and_match_counts_are_marked', async () => {
        const wrapper = mount(EventExplorerView, {
            props: { sessionId: 3 },
            global: { stubs: { RouterLink: { template: '<a><slot /></a>' } } },
        })
        await flushPromises()

        expect(wrapper.find('h2.events-heading').text()).toBe('Device-scored events')

        const card = (label: string) =>
            wrapper.findAll('.stat-card').find((c) => c.find('.stat-label').text() === label)!
        const markOf = (label: string) =>
            card(label).find('button.provenance-mark').attributes('aria-label') ?? null

        expect(markOf('SNORE-detected')).toMatch(/^Experimental/)
        expect(card('Device-scored').find('button.provenance-mark').exists()).toBe(false)
    })
})
