import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import { createMemoryHistory, createRouter, type Router } from 'vue-router'

vi.mock('@/api/stats', () => ({ getSummary: vi.fn(), getTrends: vi.fn() }))
vi.mock('@/api/days', () => ({ getDays: vi.fn() }))
vi.mock('@/api/sessions', () => ({ getSessions: vi.fn() }))
vi.mock('@/api/health', () => ({ getHealthNights: vi.fn() }))
vi.mock('@/components/TrendChart.vue', () => ({
    default: { template: '<canvas class="trend-chart-stub" />' },
}))

import { getSummary, getTrends } from '@/api/stats'
import { getDays } from '@/api/days'
import { getSessions } from '@/api/sessions'
import { getHealthNights } from '@/api/health'
import DashboardView from '@/views/DashboardView.vue'
import { setMediaMatches } from './matchMedia'

const SUMMARY = {
    days_with_data: 30,
    avg_ahi: 2.4,
    effectiveness: 'good',
    ahi_trend_direction: 'improving',
    avg_hours: 7.1,
    avg_leak: 4.2,
    event_counts: [],
}

const TRENDS = {
    ahi: [['2026-09-01', 2.1]],
    usage: [['2026-09-01', 7]],
    spo2: [['2026-09-01', 95]],
    leak: [['2026-09-01', 4]],
}

const SESSIONS = {
    items: [
        { id: 11, therapy_day: '2026-09-30', duration_hours: 7.2, ahi: 1.9 },
        { id: 12, therapy_day: '2026-09-29', duration_hours: 6.8, ahi: 2.7 },
    ],
    total: 2,
    limit: 5,
    offset: 0,
}

const HEALTH_NIGHTS = {
    items: [{ night_date: '2026-09-30', total_sleep_seconds: 25200, sleep_efficiency_pct: 90 }],
    total: 1,
    limit: 30,
    offset: 0,
}

const Stub = { template: '<div />' }

function marks(el: Element, tier: 'Derived' | 'Experimental' = 'Derived'): Element[] {
    return [...el.querySelectorAll('button.provenance-mark')].filter((b) =>
        b.getAttribute('aria-label')?.startsWith(tier),
    )
}

async function mountView(): Promise<{ wrapper: VueWrapper; router: Router }> {
    const router = createRouter({
        history: createMemoryHistory(),
        routes: [
            { path: '/', name: 'dashboard', component: Stub },
            { path: '/sessions/:id', name: 'session-detail', component: Stub },
            { path: '/days/:date', name: 'day-detail', component: Stub },
        ],
    })
    await router.push('/')
    const wrapper = mount(DashboardView, { global: { plugins: [router] }, attachTo: document.body })
    await flushPromises()
    return { wrapper, router }
}

function section(wrapper: VueWrapper, heading: string) {
    const card = wrapper
        .findAll('.section-card')
        .find((c) => c.find('h2').text().startsWith(heading))
    expect(card, `section "${heading}" should render`).toBeDefined()
    return card!
}

describe('DashboardView provenance', () => {
    beforeEach(() => {
        vi.mocked(getSummary).mockResolvedValue(SUMMARY as never)
        vi.mocked(getTrends).mockResolvedValue(TRENDS as never)
        vi.mocked(getDays).mockResolvedValue({ items: [], total: 0, limit: 365, offset: 0 })
        vi.mocked(getSessions).mockResolvedValue(SESSIONS as never)
        vi.mocked(getHealthNights).mockResolvedValue(HEALTH_NIGHTS as never)
    })

    afterEach(() => {
        setMediaMatches(false)
        document.body.innerHTML = ''
    })

    it('test_ahi_trend_badge_and_chart_heading_are_marked_derived', async () => {
        const { wrapper } = await mountView()

        expect(marks(wrapper.find('.trend-badge').element)).toHaveLength(1)
        expect(marks(section(wrapper, 'AHI Trend').find('h2').element)).toHaveLength(1)
        wrapper.unmount()
    })

    it('test_avg_sleep_cards_are_marked_derived', async () => {
        const { wrapper } = await mountView()

        for (const label of ['Avg Sleep', 'Avg Sleep Efficiency']) {
            const card = wrapper
                .findAll('.stat-card')
                .find((c) => c.find('.stat-label').text() === label)
            expect(card, `${label} card should render`).toBeDefined()
            expect(marks(card!.find('.stat-label').element)).toHaveLength(1)
        }
        wrapper.unmount()
    })

    it('test_desktop_recent_sessions_ahi_header_is_marked_derived', async () => {
        const { wrapper } = await mountView()

        const ahiHeader = section(wrapper, 'Recent Sessions')
            .findAll('th')
            .find((th) => th.text().startsWith('AHI'))
        expect(marks(ahiHeader!.element)).toHaveLength(1)
        wrapper.unmount()
    })

    it('test_mobile_recent_session_cards_mark_ahi_outside_the_link', async () => {
        setMediaMatches(true)
        const { wrapper } = await mountView()

        const list = section(wrapper, 'Recent Sessions').find('.card-list').element
        expect(list.querySelectorAll('.data-card')).toHaveLength(2)
        expect(marks(list)).toHaveLength(2)
        expect(list.querySelectorAll('a button, a .provenance-mark')).toHaveLength(0)
        wrapper.unmount()
    })

    it('test_mobile_mark_tap_does_not_navigate', async () => {
        setMediaMatches(true)
        const { wrapper, router } = await mountView()

        const list = section(wrapper, 'Recent Sessions').find('.card-list').element
        ;(marks(list)[0] as HTMLElement).click()
        await flushPromises()
        expect(router.currentRoute.value.fullPath).toBe('/')
        wrapper.unmount()
    })

    it('test_mobile_card_date_links_to_session', async () => {
        setMediaMatches(true)
        const { wrapper, router } = await mountView()

        const link = section(wrapper, 'Recent Sessions').find('.card-list .data-card-header a')
        await link.trigger('click')
        await flushPromises()
        expect(router.currentRoute.value.fullPath).toBe('/sessions/11')
        wrapper.unmount()
    })
})
