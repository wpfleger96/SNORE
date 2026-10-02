import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { createMemoryHistory, createRouter, type Router } from 'vue-router'

vi.mock('@/api/health', () => ({ getHealthNights: vi.fn() }))

import { getHealthNights } from '@/api/health'
import AppleHealthView from '@/views/AppleHealthView.vue'
import { setMediaMatches } from './matchMedia'

const NIGHTS = {
    items: [
        {
            night_date: '2026-04-06',
            total_sleep_seconds: 25200,
            sleep_efficiency_pct: 91.2,
            core_seconds: 14400,
            deep_seconds: 3600,
            rem_seconds: 7200,
            preferred_source: 'Apple Watch',
        },
        {
            night_date: '2026-04-05',
            total_sleep_seconds: 21600,
            sleep_efficiency_pct: 88.0,
            core_seconds: 12600,
            deep_seconds: 3000,
            rem_seconds: 6000,
            preferred_source: 'Apple Watch',
        },
    ],
    total: 2,
    limit: 30,
    offset: 0,
}

const Stub = { template: '<div />' }

function derivedMarks(el: Element): Element[] {
    return [...el.querySelectorAll('button.provenance-mark')].filter((b) =>
        b.getAttribute('aria-label')?.startsWith('Derived'),
    )
}

async function mountView(): Promise<{ wrapper: ReturnType<typeof mount>; router: Router }> {
    const router = createRouter({
        history: createMemoryHistory(),
        routes: [
            { path: '/apple-health', component: Stub },
            { path: '/apple-health/:date', component: Stub },
            { path: '/import', component: Stub },
        ],
    })
    await router.push('/apple-health')
    const wrapper = mount(AppleHealthView, {
        global: { plugins: [router] },
        attachTo: document.body,
    })
    await flushPromises()
    return { wrapper, router }
}

describe('AppleHealthView provenance', () => {
    beforeEach(() => {
        vi.mocked(getHealthNights).mockResolvedValue(NIGHTS as never)
    })

    afterEach(() => {
        setMediaMatches(false)
        document.body.innerHTML = ''
    })

    it('test_desktop_table_headers_mark_sleep_metrics_derived', async () => {
        const { wrapper } = await mountView()
        const headers = wrapper.findAll('th').map((th) => th.element)
        const marked = headers
            .filter((th) => derivedMarks(th).length === 1)
            .map((th) => th.textContent?.trim())
        expect(marked).toEqual(['Total Sleep', 'Efficiency', 'Core', 'Deep', 'REM'])
        wrapper.unmount()
    })

    it('test_mobile_cards_show_derived_marks_on_each_metric', async () => {
        setMediaMatches(true)
        const { wrapper } = await mountView()
        const cards = wrapper.findAll('.card-list .data-card')
        expect(cards).toHaveLength(2)
        for (const card of cards) {
            const labels = card
                .findAll('.data-card-label')
                .filter((l) => derivedMarks(l.element).length === 1)
                .map((l) => l.text())
            expect(labels).toEqual([
                'Total Sleep (hr)',
                'Efficiency (%)',
                'Core (hr)',
                'Deep (hr)',
                'REM (hr)',
            ])
        }
        wrapper.unmount()
    })

    it('test_mobile_card_marks_are_not_inside_links', async () => {
        setMediaMatches(true)
        const { wrapper } = await mountView()
        const list = wrapper.find('.card-list').element
        expect(derivedMarks(list)).toHaveLength(10)
        expect(list.querySelectorAll('a button, a .provenance-mark')).toHaveLength(0)
        wrapper.unmount()
    })

    it('test_mobile_mark_tap_does_not_navigate', async () => {
        setMediaMatches(true)
        const { wrapper, router } = await mountView()
        const mark = derivedMarks(wrapper.find('.card-list').element)[0] as HTMLElement
        mark.click()
        await flushPromises()
        expect(router.currentRoute.value.fullPath).toBe('/apple-health')
        wrapper.unmount()
    })

    it('test_mobile_card_date_links_to_night', async () => {
        setMediaMatches(true)
        const { wrapper, router } = await mountView()
        await wrapper.find('.card-list .data-card a').trigger('click')
        await flushPromises()
        expect(router.currentRoute.value.fullPath).toBe('/apple-health/2026-04-06')
        wrapper.unmount()
    })
})
