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

function experimentalMarks(el: Element): Element[] {
    return [...el.querySelectorAll('button.provenance-mark')].filter((b) =>
        b.getAttribute('aria-label')?.startsWith('Experimental'),
    )
}

async function mountView() {
    const wrapper = mount(AnalysisView, {
        props: { sessionId: 7 },
        global: { stubs: { RouterLink: { template: '<a><slot /></a>' } } },
    })
    await flushPromises()
    return wrapper
}

describe('AnalysisView provenance', () => {
    beforeEach(() => {
        vi.mocked(useAuth).mockReturnValue(makeAuthMock({ canWrite: ref(true) }) as never)
        vi.mocked(getAnalysis).mockResolvedValue(ANALYSIS as never)
        vi.mocked(getWaveformCompare).mockResolvedValue(COMPARISON as never)
    })

    afterEach(() => {
        setMediaMatches(false)
    })

    it('test_page_shows_experimental_banner', async () => {
        const wrapper = await mountView()
        const banner = wrapper.find('[role="note"]')
        expect(banner.exists()).toBe(true)
        expect(banner.text()).toContain('Experimental analysis.')
    })

    it('test_mode_comparison_ahi_header_is_marked_experimental', async () => {
        const wrapper = await mountView()
        const ahiHeader = wrapper.findAll('th').find((th) => th.text().startsWith('AHI'))
        expect(ahiHeader).toBeDefined()
        expect(experimentalMarks(ahiHeader!.element)).toHaveLength(1)
    })

    it('test_comparison_duration_mark_follows_event_source', async () => {
        const wrapper = await mountView()
        const sections = wrapper.findAll('.compare-table-section')
        const fnSection = sections.find((s) => s.text().includes('False Negatives'))!
        const fpSection = sections.find((s) => s.text().includes('False Positives'))!

        // Device-scored (machine) event durations stay unmarked.
        const fnDuration = fnSection.findAll('td').find((td) => td.text().startsWith('12.0s'))!
        expect(fnDuration.find('button.provenance-mark').exists()).toBe(false)

        // SNORE-detected (programmatic) event durations carry the experimental mark.
        const fpDuration = fpSection.findAll('td').find((td) => td.text().startsWith('15.0s'))!
        expect(experimentalMarks(fpDuration.element)).toHaveLength(1)
    })

    it('test_false_negatives_heading_names_device_scored_events', async () => {
        const wrapper = await mountView()
        expect(wrapper.text()).toContain('False Negatives: device-scored events SNORE missed')
    })
})
