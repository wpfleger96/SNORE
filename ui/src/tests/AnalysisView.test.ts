import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import { ref } from 'vue'
import { AxiosError } from 'axios'

vi.mock('@/composables/useAuth')
vi.mock('@/api/analysis', () => ({ getAnalysis: vi.fn(), runAnalysis: vi.fn() }))
vi.mock('@/api/waveforms', () => ({ getWaveformCompare: vi.fn() }))

import AnalysisView from '@/views/AnalysisView.vue'
import { useAuth } from '@/composables/useAuth'
import { getAnalysis, runAnalysis } from '@/api/analysis'
import { getWaveformCompare } from '@/api/waveforms'
import { makeAuthMock } from './helpers/mockUseAuth'
import { setMediaMatches } from './matchMedia'
import { advancedTrigger, clickMode, RERA_GROUP } from './helpers/modePicker'

const ANALYSIS = {
    session_id: 7,
    session_duration_hours: 7.5,
    total_breaths: 6000,
    machine_events: [],
    pulse_change_count: null,
    mode_results: {
        aasm: {
            ahi: 4.2,
            rdi: 6.1,
            apneas: [],
            hypopneas: [],
            reras: [],
            apnea_count: 0,
            hypopnea_count: 0,
            rera_count: 0,
        },
    },
    flow_analysis: null,
    csr_detection: null,
    periodic_breathing: null,
}

const TWO_MODE_ANALYSIS = {
    ...ANALYSIS,
    mode_results: {
        ...ANALYSIS.mode_results,
        resmed: {
            ahi: 3.1,
            rdi: 3.1,
            apneas: [],
            hypopneas: [],
            reras: [],
            apnea_count: 0,
            hypopnea_count: 0,
            rera_count: 0,
        },
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
    false_positives_rera: [],
}

function httpError(status: number) {
    return new AxiosError('failed', undefined, undefined, undefined, { status } as never)
}

const mounted: VueWrapper[] = []

// Attached so the teleported run dialog is reachable through document.body.
async function mountView() {
    const wrapper = mount(AnalysisView, {
        props: { sessionId: 7 },
        attachTo: document.body,
        global: { stubs: { RouterLink: { template: '<a><slot /></a>' } } },
    })
    mounted.push(wrapper)
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
        for (const wrapper of mounted.splice(0)) wrapper.unmount()
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

    it('test_mode_table_and_flow_section_read_typed_counts', async () => {
        vi.mocked(getAnalysis).mockResolvedValue({
            ...ANALYSIS,
            mode_results: {
                aasm: {
                    ...ANALYSIS.mode_results.aasm,
                    apnea_count: 12,
                    hypopnea_count: 34,
                    rera_count: 56,
                },
            },
            flow_analysis: {
                total_breaths: 4321,
                class_distribution: { '1': 4000, '4': 321 },
                flow_limitation_index: 0.25,
                average_confidence: 0.8,
                patterns: [],
            },
        } as never)
        const wrapper = await mountView()
        const rows = wrapper.findAll('tr').map((r) => r.findAll('td').map((c) => c.text()))
        const modeRow = rows.find((cells) => cells[0] === 'AASM')
        expect(modeRow?.slice(3)).toEqual(['12', '34', '56'])
        const class4 = rows.find((cells) => cells[1] === '4')
        expect(class4?.slice(4)).toEqual(['321', '7.4%'])
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

    function dialogButton(label: string): HTMLButtonElement {
        const button = [...document.body.querySelectorAll('button')].find(
            (b) => b.textContent?.trim() === label,
        )
        expect(button, `button "${label}"`).toBeDefined()
        return button!
    }

    async function openRunDialog(wrapper: View) {
        await wrapper
            .findAll('button')
            .find((b) => b.text() === 'Run Analysis')!
            .trigger('click')
        await flushPromises()
    }

    async function runFromDialog(wrapper: View, setUp?: () => Promise<void>) {
        await openRunDialog(wrapper)
        await setUp?.()
        dialogButton('Run').click()
        await flushPromises()
    }

    function saveResultsCheckbox(): HTMLInputElement {
        const label = [...document.body.querySelectorAll('label')].find(
            (l) => l.textContent?.trim() === 'Save results',
        )
        const input = label?.querySelector<HTMLInputElement>('input[type="checkbox"]')
        expect(input, 'Save results checkbox').toBeDefined()
        return input!
    }

    async function toggleSaving() {
        saveResultsCheckbox().click()
        await flushPromises()
    }

    function modeRowAhi(wrapper: View, mode: string) {
        const row = wrapper
            .findAll('tr')
            .map((r) => r.findAll('td').map((c) => c.text()))
            .find((cells) => cells[0] === mode)
        return row?.[1]
    }

    const PREVIEW = {
        ...ANALYSIS,
        mode_results: { resmed: { ...TWO_MODE_ANALYSIS.mode_results.resmed, ahi: 9.9 } },
    }

    it('test_running_analysis_from_the_page_shows_event_comparison', async () => {
        vi.mocked(getAnalysis).mockRejectedValue({ response: { status: 404 } })
        vi.mocked(runAnalysis).mockResolvedValue(ANALYSIS as never)
        const wrapper = await mountView()
        expect(getWaveformCompare).not.toHaveBeenCalled()

        await runFromDialog(wrapper)

        expect(runAnalysis).toHaveBeenCalledWith(7, {
            modes: ['aasm'],
            primary_mode: 'aasm',
            store_results: true,
        })
        expect(statValue(wrapper, 'Matched')).toBe('2')
    })

    it('test_run_sends_selected_modes_and_primary_mode', async () => {
        vi.mocked(runAnalysis).mockResolvedValue(TWO_MODE_ANALYSIS as never)
        const wrapper = await mountView()

        await runFromDialog(wrapper, async () => {
            await clickMode(document.body, 'Modes', 'ResMed')
            await clickMode(document.body, RERA_GROUP, 'ResMed')
        })

        expect(runAnalysis).toHaveBeenCalledWith(7, {
            modes: ['aasm', 'resmed'],
            primary_mode: 'resmed',
            store_results: true,
        })
    })

    it('test_run_is_disabled_with_no_modes', async () => {
        const wrapper = await mountView()
        await openRunDialog(wrapper)

        await clickMode(document.body, 'Modes', 'AASM')
        dialogButton('Run').click()
        await flushPromises()

        expect(dialogButton('Run').disabled).toBe(true)
        expect(runAnalysis).not.toHaveBeenCalled()
    })

    it('test_unsaved_run_shows_a_preview_and_keeps_saved_results', async () => {
        vi.mocked(runAnalysis).mockResolvedValue(PREVIEW as never)
        const wrapper = await mountView()
        expect(getWaveformCompare).toHaveBeenCalledOnce()

        await runFromDialog(wrapper, async () => {
            await clickMode(document.body, 'Modes', 'ResMed')
            await clickMode(document.body, 'Modes', 'AASM')
            await toggleSaving()
        })

        expect(runAnalysis).toHaveBeenCalledWith(7, {
            modes: ['resmed'],
            primary_mode: 'resmed',
            store_results: false,
        })
        expect(wrapper.find('[role="status"]').text()).toContain('Showing an unsaved preview.')
        expect(modeRowAhi(wrapper, 'ResMed')).toBe('9.9')
        expect(modeRowAhi(wrapper, 'AASM')).toBeUndefined()
        expect(comparisonSection(wrapper)!.text()).toContain(
            'Event comparison is available for saved results only.',
        )
        expect(comparisonSection(wrapper)!.find('.stat-card').exists()).toBe(false)

        await wrapper
            .findAll('[role="status"] button')
            .find((b) => b.text() === 'Saved results')!
            .trigger('click')
        await flushPromises()

        expect(wrapper.find('[role="status"]').text()).toContain('Showing saved results.')
        expect(modeRowAhi(wrapper, 'AASM')).toBe('4.2')
        expect(statValue(wrapper, 'Matched')).toBe('2')
        // The saved comparison is served from cache: the preview never invalidated it.
        expect(getWaveformCompare).toHaveBeenCalledOnce()
    })

    it('test_unsaved_run_without_saved_analysis_is_labelled_and_discardable', async () => {
        vi.mocked(getAnalysis).mockRejectedValue({ response: { status: 404 } })
        vi.mocked(runAnalysis).mockResolvedValue(PREVIEW as never)
        const wrapper = await mountView()

        await runFromDialog(wrapper, toggleSaving)

        const notice = wrapper.find('[role="status"]')
        expect(notice.text()).toContain('Showing an unsaved preview.')
        expect(notice.text()).toContain('This session has no saved analysis.')
        expect(notice.findAll('button').map((b) => b.text())).toEqual(['Discard preview'])
        expect(modeRowAhi(wrapper, 'ResMed')).toBe('9.9')
        expect(getWaveformCompare).not.toHaveBeenCalled()

        await notice.find('button').trigger('click')
        await flushPromises()

        expect(wrapper.find('.empty-card').text()).toContain('No analysis results')
    })

    it('test_saved_run_after_a_preview_replaces_results_and_drops_the_preview', async () => {
        vi.mocked(runAnalysis).mockResolvedValueOnce(PREVIEW as never)
        const wrapper = await mountView()
        await runFromDialog(wrapper, toggleSaving)
        vi.mocked(runAnalysis).mockResolvedValueOnce(TWO_MODE_ANALYSIS as never)

        // The dialog keeps its last choices, so saving is turned back on.
        await runFromDialog(wrapper, toggleSaving)

        expect(vi.mocked(runAnalysis).mock.calls[1]![1]).toMatchObject({ store_results: true })
        expect(wrapper.find('[role="status"]').exists()).toBe(false)
        expect(modeRowAhi(wrapper, 'ResMed')).toBe('3.1')
        // A saved run clears the comparison cache, so the comparison is fetched again.
        expect(getWaveformCompare).toHaveBeenCalledTimes(2)
    })

    it('test_failed_run_keeps_the_page_and_shows_the_error_in_the_dialog', async () => {
        vi.mocked(runAnalysis).mockRejectedValue(new Error('primary_mode must be in modes'))
        const wrapper = await mountView()

        await runFromDialog(wrapper)

        expect(document.body.textContent).toContain('primary_mode must be in modes')
        expect(modeRowAhi(wrapper, 'AASM')).toBe('4.2')
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
        expect(comparisonSection(wrapper)!.text()).toContain('Detection mode: AASM')
        expect(statValue(wrapper, 'Matched')).toBe('2')

        await selectMode(wrapper, 'ResMed')

        expect(getWaveformCompare).toHaveBeenLastCalledWith(7, { mode: 'resmed' })
        expect(comparisonSection(wrapper)!.text()).toContain('Detection mode: ResMed')
        expect(statValue(wrapper, 'Matched')).toBe('1')
    })

    it('test_mode_without_comparison_hides_the_section', async () => {
        vi.mocked(getAnalysis).mockResolvedValue(TWO_MODE_ANALYSIS as never)
        const wrapper = await mountView()
        vi.mocked(getWaveformCompare).mockRejectedValue(httpError(404))

        await selectMode(wrapper, 'ResMed')

        expect(comparisonSection(wrapper)).toBeUndefined()
        expect(wrapper.find('.error-state').exists()).toBe(false)
    })

    it('test_comparison_failure_shows_error_with_retry', async () => {
        vi.mocked(getWaveformCompare).mockRejectedValueOnce(httpError(500))
        const wrapper = await mountView()
        expect(comparisonSection(wrapper)).toBeUndefined()
        expect(wrapper.find('.error-state').text()).toContain('failed')

        await wrapper.find('.error-state button').trigger('click')
        await flushPromises()

        expect(wrapper.find('.error-state').exists()).toBe(false)
        expect(statValue(wrapper, 'Matched')).toBe('2')
    })

    it('test_comparison_shows_loading_state_until_it_arrives', async () => {
        let resolve!: (value: unknown) => void
        vi.mocked(getWaveformCompare).mockImplementation(
            (() => new Promise((r) => (resolve = r))) as never,
        )
        const wrapper = await mountView()
        expect(wrapper.text()).toContain('Loading event comparison')
        expect(comparisonSection(wrapper)).toBeUndefined()

        resolve(COMPARISON)
        await flushPromises()

        expect(wrapper.text()).not.toContain('Loading event comparison')
        expect(comparisonSection(wrapper)).toBeDefined()
    })

    it('test_returning_to_a_mode_reuses_its_cached_comparison', async () => {
        vi.mocked(getAnalysis).mockResolvedValue(TWO_MODE_ANALYSIS as never)
        const wrapper = await mountView()
        await selectMode(wrapper, 'ResMed')
        expect(getWaveformCompare).toHaveBeenCalledTimes(2)

        await selectMode(wrapper, 'AASM')

        expect(getWaveformCompare).toHaveBeenCalledTimes(2)
        expect(comparisonSection(wrapper)).toBeDefined()
    })

    it('test_false_negatives_render_in_time_order', async () => {
        vi.mocked(getWaveformCompare).mockResolvedValue({
            ...COMPARISON,
            false_negatives: [
                { event_type: 'OA', start_time: 300, duration: 10, source: 'machine' },
                { event_type: 'RE', start_time: 100, duration: 10, source: 'machine' },
                { event_type: 'CA', start_time: 200, duration: 10, source: 'machine' },
            ],
        } as never)
        const wrapper = await mountView()

        const types = compareTable(wrapper, 'False Negatives')
            .findAll('tbody tr')
            .map((row) => row.find('.event-badge').text())
        expect(types).toEqual(['RE', 'CA', 'OA'])
    })

    it('test_false_negative_without_device_duration_shows_dash_not_zero_seconds', async () => {
        vi.mocked(getWaveformCompare).mockResolvedValue({
            ...COMPARISON,
            false_negatives: [
                { event_type: 'H', start_time: 100, duration: 0, source: 'machine' },
                { event_type: 'OA', start_time: 200, duration: 12.5, source: 'machine' },
            ],
        } as never)
        const wrapper = await mountView()

        const durations = compareTable(wrapper, 'False Negatives')
            .findAll('tbody tr')
            .map((row) => row.findAll('td').at(-1)?.text())
        expect(durations).toEqual(['—', '12.5s'])
    })

    it('test_rera_false_positives_are_listed_and_counted', async () => {
        vi.mocked(getWaveformCompare).mockResolvedValue({
            ...COMPARISON,
            false_positives_rera: [
                { event_type: 'RE', start_time: 50, duration: 10, source: 'programmatic' },
            ],
        } as never)
        const wrapper = await mountView()

        expect(statValue(wrapper, 'False Positives')).toBe('2')
        const types = compareTable(wrapper, 'False Positives')
            .findAll('tbody tr')
            .map((row) => row.find('.event-badge').text())
        expect(types).toEqual(['RE', 'CA'])
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
        await selectMode(wrapper, 'ResMed')

        responses[1]!('resmed')
        await flushPromises()
        responses[0]!('aasm')
        await flushPromises()

        expect(comparisonSection(wrapper)!.text()).toContain('Detection mode: ResMed')
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
        expect(compareTable(wrapper, heading).text()).toContain('1–25 of 30')

        await clickNext(wrapper, heading)

        expect(compareTable(wrapper, heading).findAll('tbody tr')).toHaveLength(5)
        expect(compareTable(wrapper, heading).text()).toContain('26–30 of 30')
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
        vi.mocked(getWaveformCompare).mockImplementation((async (
            _sessionId: number,
            params?: { mode?: string },
        ) => ({
            ...COMPARISON,
            mode: params?.mode,
            false_negatives: falseEvents(30),
        })) as never)
        const wrapper = await mountView()
        await clickNext(wrapper, 'False Negatives')

        await selectMode(wrapper, 'ResMed')

        expect(compareTable(wrapper, 'False Negatives').text()).toContain('1–25 of 30')
    })

    function runDialog() {
        return document.body.querySelector('[role="alertdialog"]')
    }

    function pressEscape() {
        document.dispatchEvent(new KeyboardEvent('keydown', { key: 'Escape', bubbles: true }))
    }

    function deferred<T = unknown>() {
        let resolve!: (value: T) => void
        let reject!: (reason: unknown) => void
        const promise = new Promise<T>((res, rej) => {
            resolve = res
            reject = rej
        })
        return { promise, resolve, reject }
    }

    it('test_explicit_primary_pick_survives_reopening_the_run_dialog', async () => {
        vi.mocked(runAnalysis).mockResolvedValue(TWO_MODE_ANALYSIS as never)
        const wrapper = await mountView()
        await openRunDialog(wrapper)
        await clickMode(document.body, 'Modes', 'ResMed')
        await clickMode(document.body, RERA_GROUP, 'ResMed')
        dialogButton('Cancel').click()
        await flushPromises()
        expect(runDialog()).toBeNull()

        await runFromDialog(wrapper, async () => {
            expect(advancedTrigger(document.body)!.textContent!.trim()).toBe(
                'Advanced: RERA flags from ResMed',
            )
        })

        expect(runAnalysis).toHaveBeenCalledWith(7, {
            modes: ['aasm', 'resmed'],
            primary_mode: 'resmed',
            store_results: true,
        })
    })

    it('test_reclicked_default_primary_stays_pinned_when_aasm_is_added', async () => {
        vi.mocked(runAnalysis).mockResolvedValue(TWO_MODE_ANALYSIS as never)
        const wrapper = await mountView()
        await openRunDialog(wrapper)
        await clickMode(document.body, 'Modes', 'AASM Relaxed')
        await clickMode(document.body, 'Modes', 'ResMed')
        await clickMode(document.body, 'Modes', 'AASM')
        await clickMode(document.body, RERA_GROUP, 'AASM Relaxed')
        dialogButton('Cancel').click()
        await flushPromises()

        await runFromDialog(wrapper, () => clickMode(document.body, 'Modes', 'AASM'))

        expect(runAnalysis).toHaveBeenCalledWith(7, {
            modes: ['aasm', 'aasm_relaxed', 'resmed'],
            primary_mode: 'aasm_relaxed',
            store_results: true,
        })
    })

    it('test_unsaved_run_hides_the_stored_rera_choice_but_sends_a_primary', async () => {
        vi.mocked(runAnalysis).mockResolvedValue(TWO_MODE_ANALYSIS as never)
        const wrapper = await mountView()
        await openRunDialog(wrapper)
        await clickMode(document.body, 'Modes', 'ResMed')
        expect(saveResultsCheckbox().checked).toBe(true)
        expect(advancedTrigger(document.body)).not.toBeNull()

        await toggleSaving()

        expect(saveResultsCheckbox().checked).toBe(false)
        expect(advancedTrigger(document.body)).toBeNull()
        dialogButton('Run').click()
        await flushPromises()
        expect(runAnalysis).toHaveBeenCalledWith(7, {
            modes: ['aasm', 'resmed'],
            primary_mode: 'aasm',
            store_results: false,
        })
    })

    it('test_run_dialog_stays_open_while_running_and_shows_the_failure', async () => {
        const run = deferred()
        vi.mocked(runAnalysis).mockReturnValue(run.promise as never)
        const wrapper = await mountView()

        await runFromDialog(wrapper)
        pressEscape()
        await flushPromises()

        expect(runDialog()).not.toBeNull()
        run.reject(new Error('analysis failed'))
        await flushPromises()
        expect(runDialog()!.textContent).toContain('analysis failed')

        // Once the run has ended, Escape closes the dialog again.
        pressEscape()
        await flushPromises()
        expect(runDialog()).toBeNull()
    })

    const RELAXED_PREVIEW = {
        ...ANALYSIS,
        mode_results: {
            aasm_relaxed: {
                ...ANALYSIS.mode_results.aasm,
                ahi: 8.8,
                apnea_count: 1,
                apneas: [
                    {
                        event_type: 'CA',
                        start_time: 60,
                        duration: 12,
                        flow_reduction: 0.9,
                        confidence: 0.8,
                    },
                ],
            },
        },
    }

    function compareByMode() {
        vi.mocked(getWaveformCompare).mockImplementation((async (
            _sessionId: number,
            params?: { mode?: string },
        ) => ({ ...COMPARISON, mode: params?.mode })) as never)
    }

    async function runRelaxedPreview(wrapper: View) {
        vi.mocked(runAnalysis).mockResolvedValue(RELAXED_PREVIEW as never)
        await runFromDialog(wrapper, async () => {
            await clickMode(document.body, 'Modes', 'AASM Relaxed')
            await clickMode(document.body, 'Modes', 'AASM')
            await toggleSaving()
        })
    }

    function eventsByMode(wrapper: View) {
        return wrapper
            .findAll('.section-card')
            .find((s) => s.find('h2').text() === 'Events by Mode')!
    }

    it('test_events_by_mode_follows_the_shown_results_and_restores_the_saved_pick', async () => {
        vi.mocked(getAnalysis).mockResolvedValue(TWO_MODE_ANALYSIS as never)
        compareByMode()
        const wrapper = await mountView()
        await selectMode(wrapper, 'ResMed')
        expect(getWaveformCompare).toHaveBeenCalledTimes(2)

        await runRelaxedPreview(wrapper)

        const section = eventsByMode(wrapper)
        expect(section.findAll('.mode-selector button').map((b) => b.text())).toEqual([
            'AASM Relaxed',
        ])
        expect(pressedModesIn(section.element)).toEqual(['AASM Relaxed'])
        expect(section.findAll('.event-badge').map((b) => b.text())).toEqual(['CA'])

        await wrapper
            .findAll('[role="status"] button')
            .find((b) => b.text() === 'Saved results')!
            .trigger('click')
        await flushPromises()

        expect(pressedModesIn(eventsByMode(wrapper).element)).toEqual(['ResMed'])
        expect(comparisonSection(wrapper)!.text()).toContain('Detection mode: ResMed')
        expect(getWaveformCompare).toHaveBeenCalledTimes(2)
    })

    function pressedModesIn(root: Element) {
        return [...root.querySelectorAll('.mode-selector button[data-state="on"]')].map((b) =>
            b.textContent!.trim(),
        )
    }

    it('test_preview_notice_and_mode_comparison_use_mode_labels', async () => {
        const wrapper = await mountView()

        await runRelaxedPreview(wrapper)

        const notice = wrapper.find('[role="status"]').text()
        expect(notice).toContain('(AASM Relaxed)')
        expect(notice).not.toContain('aasm_relaxed')
        expect(modeRowAhi(wrapper, 'AASM Relaxed')).toBe('8.8')
        expect(wrapper.text()).not.toContain('aasm_relaxed')
    })

    it('test_saved_rerun_hides_the_replaced_comparison_until_the_new_one_loads', async () => {
        const wrapper = await mountView()
        expect(statValue(wrapper, 'Matched')).toBe('2')
        const next = deferred()
        vi.mocked(getWaveformCompare).mockReturnValue(next.promise as never)
        vi.mocked(runAnalysis).mockResolvedValue({ ...ANALYSIS } as never)

        await runFromDialog(wrapper)

        expect(wrapper.text()).toContain('Loading event comparison')
        expect(comparisonSection(wrapper)).toBeUndefined()
        next.resolve({ ...COMPARISON, matched_events: 3 })
        await flushPromises()
        expect(statValue(wrapper, 'Matched')).toBe('3')
    })

    it('test_comparison_for_a_replaced_analysis_is_not_cached', async () => {
        const requests: Array<ReturnType<typeof deferred>> = []
        vi.mocked(getWaveformCompare).mockImplementation((() => {
            requests.push(deferred())
            return requests.at(-1)!.promise
        }) as never)
        vi.mocked(runAnalysis).mockResolvedValue(TWO_MODE_ANALYSIS as never)
        const wrapper = await mountView()

        await runFromDialog(wrapper)
        requests[1]!.resolve({ ...COMPARISON, matched_events: 5 })
        await flushPromises()
        // The request made for the replaced analysis lands last.
        requests[0]!.resolve({ ...COMPARISON, matched_events: 2 })
        await flushPromises()
        expect(statValue(wrapper, 'Matched')).toBe('5')

        await selectMode(wrapper, 'ResMed')
        requests[2]!.resolve({ ...COMPARISON, mode: 'resmed' })
        await flushPromises()
        await selectMode(wrapper, 'AASM')

        expect(requests).toHaveLength(3)
        expect(statValue(wrapper, 'Matched')).toBe('5')
    })

    it('test_session_change_loads_its_analysis_and_drops_the_old_run', async () => {
        const wrapper = await mountView()
        await runRelaxedPreview(wrapper)
        expect(wrapper.find('[role="status"]').exists()).toBe(true)
        const run = deferred()
        vi.mocked(runAnalysis).mockReturnValue(run.promise as never)
        await runFromDialog(wrapper)

        vi.mocked(getAnalysis).mockResolvedValue({
            ...ANALYSIS,
            session_id: 8,
            mode_results: { aasm: { ...ANALYSIS.mode_results.aasm, ahi: 1.5 } },
        } as never)
        await wrapper.setProps({ sessionId: 8 })
        await flushPromises()
        run.resolve(TWO_MODE_ANALYSIS)
        await flushPromises()

        expect(getAnalysis).toHaveBeenLastCalledWith(8)
        expect(wrapper.find('[role="status"]').exists()).toBe(false)
        expect(modeRowAhi(wrapper, 'AASM')).toBe('1.5')
        expect(modeRowAhi(wrapper, 'ResMed')).toBeUndefined()
        expect(getWaveformCompare).toHaveBeenLastCalledWith(8, { mode: 'aasm' })
    })
})
