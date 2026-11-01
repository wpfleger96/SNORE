import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { ref } from 'vue'

vi.mock('@/composables/useAuth')
vi.mock('@/api/analysis', () => ({ getAnalysis: vi.fn(), runAnalysis: vi.fn() }))
vi.mock('@/api/waveforms', () => ({ getWaveformCompare: vi.fn() }))

import AnalysisView from '@/views/AnalysisView.vue'
import { useAuth } from '@/composables/useAuth'
import { getAnalysis, runAnalysis } from '@/api/analysis'
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

const TWO_MODE_ANALYSIS = {
    ...ANALYSIS,
    mode_results: {
        ...ANALYSIS.mode_results,
        resmed: { ahi: 3.1, rdi: 3.1, apneas: [], hypopneas: [], reras: [] },
    },
}

const COMPARISON = {
    mode: 'aasm',
    machine_event_count: 3,
    programmatic_event_count: 4,
    matched_events: 2,
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

    type View = Awaited<ReturnType<typeof mountView>>

    function comparisonSection(wrapper: View) {
        return wrapper
            .findAll('.section-card')
            .find((s) => s.find('h2').text() === 'Event Comparison')
    }

    async function selectMode(wrapper: View, mode: string) {
        const item = wrapper.findAll('.mode-selector button').find((b) => b.text() === mode)!
        await item.trigger('click')
        await flushPromises()
    }

    function statValue(wrapper: View, label: string) {
        return comparisonSection(wrapper)!
            .findAll('.stat-card')
            .find((c) => c.find('.stat-label').text().startsWith(label))!
            .find('.stat-value')
            .text()
    }

    it('test_event_comparison_shows_matched_and_sensitivity', async () => {
        const wrapper = await mountView()

        expect(statValue(wrapper, 'Matched')).toBe('2')
        expect(statValue(wrapper, 'Sensitivity')).toBe('66.7 %')
    })

    it('test_sensitivity_is_empty_with_no_device_events', async () => {
        vi.mocked(getWaveformCompare).mockResolvedValue({
            ...COMPARISON,
            machine_event_count: 0,
            matched_events: 0,
        } as never)
        const wrapper = await mountView()

        expect(statValue(wrapper, 'Matched')).toBe('0')
        expect(statValue(wrapper, 'Sensitivity')).toBe('---')
    })

    it('test_running_analysis_from_the_page_shows_event_comparison', async () => {
        vi.mocked(getAnalysis).mockRejectedValue({ response: { status: 404 } })
        vi.mocked(runAnalysis).mockResolvedValue(ANALYSIS as never)
        const wrapper = await mountView()
        expect(getWaveformCompare).not.toHaveBeenCalled()

        await wrapper.find('.empty-card button').trigger('click')
        await flushPromises()

        expect(statValue(wrapper, 'Matched')).toBe('2')
    })

    it('test_comparison_follows_the_selected_mode', async () => {
        vi.mocked(getAnalysis).mockResolvedValue(TWO_MODE_ANALYSIS as never)
        vi.mocked(getWaveformCompare).mockImplementation((async (
            _sessionId: number,
            params?: { mode?: string },
        ) => ({
            ...COMPARISON,
            mode: params?.mode,
            matched_events: params?.mode === 'resmed' ? 1 : 2,
        })) as never)
        const wrapper = await mountView()
        expect(comparisonSection(wrapper)!.text()).toContain('Detection mode: aasm')
        expect(statValue(wrapper, 'Matched')).toBe('2')

        await selectMode(wrapper, 'resmed')

        expect(getWaveformCompare).toHaveBeenLastCalledWith(7, { mode: 'resmed' })
        expect(comparisonSection(wrapper)!.text()).toContain('Detection mode: resmed')
        expect(statValue(wrapper, 'Matched')).toBe('1')
    })

    it('test_mode_without_comparison_hides_the_section', async () => {
        vi.mocked(getAnalysis).mockResolvedValue(TWO_MODE_ANALYSIS as never)
        const wrapper = await mountView()
        vi.mocked(getWaveformCompare).mockRejectedValue(new Error('Mode not found'))

        await selectMode(wrapper, 'resmed')

        expect(comparisonSection(wrapper)).toBeUndefined()
    })

    it('test_slow_response_for_an_earlier_mode_is_dropped', async () => {
        vi.mocked(getAnalysis).mockResolvedValue(TWO_MODE_ANALYSIS as never)
        const responses: Array<(mode: string) => void> = []
        vi.mocked(getWaveformCompare).mockImplementation(
            (() =>
                new Promise((resolve) => {
                    responses.push((mode) => resolve({ ...COMPARISON, mode }))
                })) as never,
        )
        const wrapper = await mountView()
        await selectMode(wrapper, 'resmed')

        responses[1]!('resmed')
        await flushPromises()
        responses[0]!('aasm')
        await flushPromises()

        expect(comparisonSection(wrapper)!.text()).toContain('Detection mode: resmed')
    })

    function falseEvents(count: number) {
        return Array.from({ length: count }, (_, i) => ({
            event_type: 'OA',
            start_time: i,
            duration: 10,
            source: 'machine',
        }))
    }

    function compareTable(wrapper: View, heading: string) {
        const table = comparisonSection(wrapper)!
            .findAll('.compare-table-section')
            .find((s) => s.find('h3').text().startsWith(heading))!
        return table
    }

    async function clickNext(wrapper: View, heading: string) {
        const next = compareTable(wrapper, heading)
            .findAll('button')
            .find((b) => b.text() === 'Next')!
        await next.trigger('click')
    }

    it.each([
        ['False Negatives', 'false_negatives'],
        ['False Positives', 'false_positives_apnea'],
    ])('test_%s_table_pages_at_25_rows', async (heading, field) => {
        vi.mocked(getWaveformCompare).mockResolvedValue({
            ...COMPARISON,
            [field]: falseEvents(30),
        } as never)
        const wrapper = await mountView()
        expect(compareTable(wrapper, heading).findAll('tbody tr')).toHaveLength(25)
        expect(compareTable(wrapper, heading).text()).toContain('Page 1 of 2')

        await clickNext(wrapper, heading)

        expect(compareTable(wrapper, heading).findAll('tbody tr')).toHaveLength(5)
        expect(compareTable(wrapper, heading).text()).toContain('Page 2 of 2')
    })

    it('test_mobile_false_negative_cards_page_at_25', async () => {
        setMediaMatches(true)
        vi.mocked(getWaveformCompare).mockResolvedValue({
            ...COMPARISON,
            false_negatives: falseEvents(30),
        } as never)
        const wrapper = await mountView()

        const section = compareTable(wrapper, 'False Negatives')
        expect(section.findAll('.data-card')).toHaveLength(25)
    })

    it('test_changing_mode_returns_comparison_tables_to_the_first_page', async () => {
        vi.mocked(getAnalysis).mockResolvedValue(TWO_MODE_ANALYSIS as never)
        vi.mocked(getWaveformCompare).mockResolvedValue({
            ...COMPARISON,
            false_negatives: falseEvents(30),
        } as never)
        const wrapper = await mountView()
        await clickNext(wrapper, 'False Negatives')

        await selectMode(wrapper, 'resmed')

        expect(compareTable(wrapper, 'False Negatives').text()).toContain('Page 1 of 2')
    })
})
