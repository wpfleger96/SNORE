import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { ref } from 'vue'

vi.mock('@/composables/useAuth')
vi.mock('@/api/analysis', () => ({ getAnalysis: vi.fn(), runAnalysis: vi.fn() }))
vi.mock('@/api/waveforms', () => ({ getWaveformCompare: vi.fn() }))

import AnalysisView from '@/views/AnalysisView.vue'
import { useAuth } from '@/composables/useAuth'
import { getAnalysis } from '@/api/analysis'
import { getWaveformCompare } from '@/api/waveforms'
import { makeAuthMock } from './helpers/mockUseAuth'
import { setMediaMatches } from './matchMedia'

const ANALYSIS = {
    session_id: 7,
    session_duration_hours: 7.5,
    total_breaths: 6000,
    machine_events: [],
    pulse_change_count: null,
    mode_results: {
        aasm: { ahi: 4.2, rdi: 6.1, apneas: [], hypopneas: [], reras: [] },
    },
    flow_analysis: null,
    csr_detection: null,
    periodic_breathing: null,
}

const COMPARISON = {
    machine_event_count: 3,
    programmatic_event_count: 4,
    false_negatives: [{ event_type: 'OA', start_time: 100, duration: 12, source: 'machine' }],
    false_positives_apnea: [
        { event_type: 'CA', start_time: 200, duration: 15, source: 'programmatic' },
    ],
    false_positives_hypopnea: [],
}

async function mountView() {
    const wrapper = mount(AnalysisView, {
        props: { sessionId: 7 },
        global: { stubs: { RouterLink: { template: '<a><slot /></a>' } } },
    })
    await flushPromises()
    return wrapper
}

describe('AnalysisView', () => {
    beforeEach(() => {
        vi.mocked(useAuth).mockReturnValue(makeAuthMock({ canWrite: ref(true) }) as never)
        vi.mocked(getAnalysis).mockResolvedValue(ANALYSIS as never)
        vi.mocked(getWaveformCompare).mockResolvedValue(COMPARISON as never)
    })

    afterEach(() => {
        setMediaMatches(false)
    })

    const MODE_AHI_HINT = 'button[aria-label="More information about Mode AHI"]'

    it('test_page_shows_one_experimental_banner', async () => {
        const wrapper = await mountView()
        const banners = wrapper.findAll('[role="note"]')
        expect(banners).toHaveLength(1)
        expect(banners[0]!.text()).toContain('Experimental analysis.')
    })

    it('test_mode_comparison_ahi_hint_explains_mode_ahi', async () => {
        const wrapper = await mountView()
        const ahiHeader = wrapper.findAll('th').find((th) => th.text().startsWith('AHI'))!
        expect(ahiHeader.find(MODE_AHI_HINT).exists()).toBe(true)
    })

    it('test_mobile_mode_card_ahi_hint_explains_mode_ahi', async () => {
        setMediaMatches(true)
        const wrapper = await mountView()
        expect(wrapper.find('table').exists()).toBe(false)

        const ahiLabel = wrapper
            .findAll('.data-card .data-card-label')
            .find((l) => l.text().startsWith('AHI'))!
        expect(ahiLabel.find(MODE_AHI_HINT).exists()).toBe(true)
    })

    it('test_event_comparison_names_device_scored_and_snore_detected', async () => {
        const wrapper = await mountView()
        const section = wrapper
            .findAll('.section-card')
            .find((s) => s.find('h2').text() === 'Event Comparison')!
        const labels = section.findAll('.stat-card .stat-label').map((l) => l.text())
        expect(labels).toEqual(
            expect.arrayContaining(['Device-scored Events', 'SNORE-detected Events']),
        )
        expect(section.text()).toContain('False Negatives: Device-scored Events SNORE Missed')
        expect(section.text()).toContain(
            'False Positives: SNORE-detected Events the Device Did Not Score',
        )
    })
})
