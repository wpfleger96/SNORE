import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { ref } from 'vue'
import { createMemoryHistory, createRouter } from 'vue-router'

vi.mock('@/composables/useAuth')
vi.mock('@/api/sessions', () => ({
    getSessions: vi.fn(),
    updateSession: vi.fn(),
    deleteSessions: vi.fn(),
    getSessionDeletePreview: vi.fn(),
    getBulkDeletePreview: vi.fn(),
}))
vi.mock('@/api/devices', () => ({ getDevices: vi.fn() }))
vi.mock('@/api/days', () => ({ getDates: vi.fn() }))
vi.mock('@/api/stats', () => ({ getDataRange: vi.fn() }))

import { getSessions } from '@/api/sessions'
import { getDevices } from '@/api/devices'
import { getDates } from '@/api/days'
import { getDataRange } from '@/api/stats'
import { useAuth } from '@/composables/useAuth'
import SessionListView from '@/views/SessionListView.vue'
import { makeAuthMock } from './helpers/mockUseAuth'
import { setMediaMatches } from './matchMedia'

const SESSIONS = {
    items: [
        {
            id: 1470,
            start_time: '2026-04-06T23:42:24',
            therapy_day: '2026-04-06',
            duration_hours: 9.59,
            ahi: 0.52,
            enabled: true,
            manufacturer: 'ResMed',
            model: 'AirSense11AutoSet',
            serial_number: '23456789012',
        },
        {
            id: 1473,
            start_time: '2026-04-06T00:21:31',
            therapy_day: '2026-04-05',
            duration_hours: 7.95,
            ahi: 0.38,
            enabled: true,
            manufacturer: 'ResMed',
            model: 'AirSense11AutoSet',
            serial_number: '23456789012',
        },
    ],
    total: 2,
    limit: 25,
    offset: 0,
}

function markTiers(el: Element): string[] {
    return [...el.querySelectorAll('button.provenance-mark')].map(
        (b) => b.getAttribute('aria-label')?.split(':')[0] ?? '',
    )
}

async function mountView() {
    const router = createRouter({
        history: createMemoryHistory(),
        routes: [
            { path: '/sessions', component: { template: '<div />' } },
            { path: '/sessions/:id', name: 'session-detail', component: { template: '<div />' } },
            {
                path: '/sessions/:id/events',
                name: 'session-events',
                component: { template: '<div />' },
            },
        ],
    })
    await router.push('/sessions')
    // SelectContent is stubbed: the device filter's empty-value "All Devices"
    // item throws in reka-ui's SelectItem setup under jsdom.
    const wrapper = mount(SessionListView, {
        global: { plugins: [router], stubs: { SelectContent: true } },
    })
    await flushPromises()
    return wrapper
}

describe('SessionListView provenance', () => {
    beforeEach(() => {
        vi.mocked(useAuth).mockReturnValue(makeAuthMock({ canWrite: ref(true) }) as never)
        vi.mocked(getSessions).mockResolvedValue(SESSIONS as never)
        vi.mocked(getDevices).mockResolvedValue([] as never)
        vi.mocked(getDates).mockRejectedValue(new Error('offline'))
        vi.mocked(getDataRange).mockRejectedValue(new Error('offline'))
    })

    afterEach(() => {
        setMediaMatches(false)
    })

    it('test_desktop_headers_mark_ahi_derived_and_duration_unmarked', async () => {
        const wrapper = await mountView()
        const tiers = Object.fromEntries(
            wrapper.findAll('th').map((th) => [th.text(), markTiers(th.element)]),
        )
        expect(tiers['AHI']).toEqual(['Derived'])
        expect(tiers['Duration']).toEqual([])
        expect(tiers['Date']).toEqual([])
    })

    it('test_mobile_cards_mark_each_session_ahi_derived', async () => {
        setMediaMatches(true)
        const wrapper = await mountView()
        const cards = wrapper.findAll('.card-list .data-card')
        expect(cards).toHaveLength(2)
        for (const card of cards) {
            expect(markTiers(card.element)).toEqual(['Derived'])
            expect(card.element.querySelectorAll('a .provenance-mark')).toHaveLength(0)
        }
    })
})
