import { beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'

vi.mock('@/api/stats', () => ({ getExperimentalStats: vi.fn() }))

// Stub InfoHint so labels render as plain text (no Popover/teleport).
vi.mock('@/components/InfoHint.vue', () => ({
    default: { template: '<span class="info-hint-stub" />' },
}))

import { getExperimentalStats } from '@/api/stats'
import TherapyAssessmentCard from '@/components/TherapyAssessmentCard.vue'

async function mountCard(): Promise<VueWrapper> {
    const wrapper = mount(TherapyAssessmentCard)
    await flushPromises()
    return wrapper
}

describe('TherapyAssessmentCard', () => {
    beforeEach(() => {
        vi.clearAllMocks()
    })

    it('test_shows_effectiveness_and_trend_under_a_banner', async () => {
        vi.mocked(getExperimentalStats).mockResolvedValue({
            effectiveness: 'good',
            ahi_trend_direction: 'worsening',
        })

        const wrapper = await mountCard()

        expect(wrapper.findAll('[role="note"]')).toHaveLength(1)
        expect(wrapper.text()).toContain('good')
        const trend = wrapper.find('.trend-badge')
        expect(trend.text()).toBe('worsening')
        expect(trend.classes()).toContain('trend-worsening')
    })

    it('test_banner_says_it_covers_all_recorded_data', async () => {
        vi.mocked(getExperimentalStats).mockResolvedValue({
            effectiveness: 'good',
            ahi_trend_direction: null,
        })

        const wrapper = await mountCard()

        expect(wrapper.find('[role="note"]').text()).toContain('all recorded data')
    })

    it('test_hides_unknown_band_but_keeps_a_real_trend', async () => {
        vi.mocked(getExperimentalStats).mockResolvedValue({
            effectiveness: 'unknown',
            ahi_trend_direction: 'stable',
        })

        const wrapper = await mountCard()

        expect(wrapper.text()).not.toContain('unknown')
        expect(wrapper.text()).not.toContain('Effectiveness')
        expect(wrapper.find('.trend-badge').text()).toBe('stable')
    })

    it('test_shows_empty_state_when_nothing_to_show', async () => {
        vi.mocked(getExperimentalStats).mockResolvedValue({
            effectiveness: 'unknown',
            ahi_trend_direction: null,
        })

        const wrapper = await mountCard()

        expect(wrapper.find('.assessment-row').exists()).toBe(false)
        expect(wrapper.find('.assessment-empty').exists()).toBe(true)
    })

    it('test_failed_load_shows_the_error_instead_of_the_empty_state', async () => {
        vi.mocked(getExperimentalStats).mockRejectedValue(new Error('boom'))

        const wrapper = await mountCard()

        expect(wrapper.find('.error-state').text()).toContain('boom')
        expect(wrapper.find('.assessment-empty').exists()).toBe(false)
    })
})
