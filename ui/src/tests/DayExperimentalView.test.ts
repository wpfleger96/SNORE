import { describe, expect, it, vi, beforeEach } from 'vitest'
import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import type { DayExperimentalMetrics } from '@/types'

vi.mock('@/api/days', () => ({
    getDayExperimental: vi.fn(),
}))

// Stub InfoHint so stat-card labels render as plain text (no Popover/teleport),
// keeping the real StatCard so its em-dash + title-tooltip logic is exercised.
vi.mock('@/components/InfoHint.vue', () => ({
    default: { template: '<span class="info-hint-stub" />' },
}))

import { getDayExperimental } from '@/api/days'
import DayExperimentalView from '@/views/DayExperimentalView.vue'

const DAY = '2026-05-03'

function makeMetrics(overrides: Partial<DayExperimentalMetrics> = {}): DayExperimentalMetrics {
    return { date: DAY, ...overrides }
}

async function mountView(): Promise<VueWrapper> {
    const wrapper = mount(DayExperimentalView, {
        props: { dayDate: DAY },
        global: { stubs: { RouterLink: { template: '<a><slot /></a>' } } },
    })
    await flushPromises()
    return wrapper
}

/** Locate a stat card by its (exact) label text; undefined when the card is not rendered. */
function findCard(wrapper: VueWrapper, label: string) {
    return wrapper.findAll('.stat-card').find((c) => c.find('.stat-label').text() === label)
}

function hasBackLink(wrapper: VueWrapper): boolean {
    return wrapper.find(`a[to="/days/${DAY}"]`).exists()
}

describe('DayExperimentalView', () => {
    beforeEach(() => {
        vi.clearAllMocks()
    })

    it('test_requests_the_routed_date', async () => {
        vi.mocked(getDayExperimental).mockResolvedValue(makeMetrics())

        await mountView()

        expect(getDayExperimental).toHaveBeenCalledWith(DAY)
    })

    it('test_date_change_refetches_and_replaces_the_shown_metrics', async () => {
        const nextDay = '2026-05-04'
        vi.mocked(getDayExperimental)
            .mockResolvedValueOnce(makeMetrics({ rera_count: 2 }))
            .mockResolvedValueOnce(makeMetrics({ date: nextDay, rera_count: 7 }))
        const wrapper = await mountView()

        await wrapper.setProps({ dayDate: nextDay })
        await flushPromises()

        expect(getDayExperimental).toHaveBeenLastCalledWith(nextDay)
        expect(findCard(wrapper, 'RERA Proxy Count')?.text()).toContain('7')
    })

    it('test_metrics_sit_under_one_experimental_banner_with_link_back', async () => {
        vi.mocked(getDayExperimental).mockResolvedValue(
            makeMetrics({ fl_class_ge4_pct: 3, rera_index: 5.5, rera_count: 2 }),
        )

        const wrapper = await mountView()

        const banners = wrapper.findAll('[role="note"]')
        expect(banners).toHaveLength(1)
        const group = banners[0]!.element.parentElement!
        expect(group.textContent).toContain('RERA Proxy Index')
        expect(group.textContent).toContain('FL Class ≥4')
        expect(group.textContent).toContain('RERA Proxy Count')
        expect(hasBackLink(wrapper)).toBe(true)
    })

    it('test_numeric_values_render_including_zero', async () => {
        vi.mocked(getDayExperimental).mockResolvedValue(
            makeMetrics({ fl_class_ge4_pct: 0, rera_index: 5.5, rera_count: 0 }),
        )

        const wrapper = await mountView()

        // 0 is a real value, not absence: the card renders rather than showing an em-dash.
        const countCard = findCard(wrapper, 'RERA Proxy Count')
        expect(countCard!.find('.stat-empty').exists()).toBe(false)
        expect(countCard!.find('.stat-value').text()).toBe('0')

        expect(findCard(wrapper, 'FL Class ≥4')!.find('.stat-value').text()).toContain('0.0')
        expect(findCard(wrapper, 'RERA Proxy Index')!.find('.stat-value').text()).toBe('5.50')
    })

    it('test_null_value_with_reason_renders_em_dash_and_tooltip', async () => {
        vi.mocked(getDayExperimental).mockResolvedValue(
            makeMetrics({
                fl_class_ge4_pct: null,
                fl_class_ge4_pct_reason: 'analysis_not_run',
                rera_index: null,
                rera_index_reason: 'analysis_not_run',
                rera_count: null,
                rera_count_reason: 'analysis_not_run',
            }),
        )

        const wrapper = await mountView()

        // Every card reads its own reason field, so check all three, not just one.
        for (const label of ['FL Class ≥4', 'RERA Proxy Index', 'RERA Proxy Count']) {
            const empty = findCard(wrapper, label)!.find('.stat-empty')
            expect(empty.text()).toBe('---')
            expect(empty.attributes('title')).toBe(
                'Breath analysis has not been run for this night.',
            )
        }
    })

    it('test_unknown_day_shows_empty_state_with_link_back', async () => {
        vi.mocked(getDayExperimental).mockRejectedValue(
            Object.assign(new Error('Not Found'), {
                isAxiosError: true,
                response: { status: 404 },
            }),
        )

        const wrapper = await mountView()

        expect(wrapper.text()).toContain('No CPAP data recorded for this date.')
        expect(wrapper.find('.error-state').exists()).toBe(false)
        expect(hasBackLink(wrapper)).toBe(true)
    })

    it('test_other_failures_show_the_error_state', async () => {
        vi.mocked(getDayExperimental).mockRejectedValue(new Error('boom'))

        const wrapper = await mountView()

        expect(wrapper.find('.error-state').text()).toContain('boom')
        expect(wrapper.text()).not.toContain('No CPAP data recorded')
    })
})
