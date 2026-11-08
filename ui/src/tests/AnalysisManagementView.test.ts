import { describe, expect, it, vi, beforeEach, afterEach } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { ref } from 'vue'

vi.mock('@/composables/useAuth')
vi.mock('@/utils/formatting', () => ({ formatDateShort: (d: string) => d }))
vi.mock('@/components/PaginationBar.vue', () => ({
    default: { template: '<div />', props: ['offset', 'pageSize', 'total'] },
}))
vi.mock('@/components/DeleteConfirmDialog.vue', () => ({
    default: {
        template: '<div />',
        props: ['visible', 'title', 'message', 'loading', 'deleting'],
    },
}))
vi.mock('@/components/AnalysisJobsBanner.vue', () => ({
    default: { template: '<div />', props: ['jobs'] },
}))
vi.mock('@/api/analysis', async (importOriginal) => {
    const actual = await importOriginal<typeof import('@/api/analysis')>()
    return {
        ...actual,
        getAnalysisSessions: vi.fn().mockResolvedValue({ items: [], total: 0 }),
        getAnalysisJobs: vi.fn().mockResolvedValue({ jobs: [] }),
        runBatchAnalysis: vi.fn().mockResolvedValue({ job_id: 'job-1', session_count: 5 }),
        deleteAnalysis: vi.fn(),
        getAnalysisDeletePreview: vi.fn(),
        cancelAnalysisJob: vi.fn(),
    }
})
vi.mock('@/api/stats', async (importOriginal) => ({
    ...(await importOriginal<typeof import('@/api/stats')>()),
    getExperimentalStats: vi.fn(),
}))

import AnalysisManagementView from '@/views/AnalysisManagementView.vue'
import DatePickerInput from '@/components/DatePickerInput.vue'
import { useAuth } from '@/composables/useAuth'
import {
    runBatchAnalysis,
    getAnalysisSessions,
    getAnalysisJobs,
    cancelAnalysisJob,
} from '@/api/analysis'
import { getExperimentalStats } from '@/api/stats'
import { makeAuthMock } from './helpers/mockUseAuth'
import { setMediaMatches } from './matchMedia'

describe('AnalysisManagementView — analyze missing', () => {
    beforeEach(() => {
        vi.resetAllMocks()
        vi.mocked(useAuth).mockReturnValue(
            makeAuthMock({ canWrite: ref(true), isAuthenticated: ref(true) }) as never,
        )
        vi.mocked(getAnalysisSessions).mockResolvedValue({
            items: [],
            total: 0,
            limit: 25,
            offset: 0,
        })
        vi.mocked(getAnalysisJobs).mockResolvedValue({ jobs: [] })
        vi.mocked(runBatchAnalysis).mockResolvedValue({ job_id: 'job-1', session_count: 5 })
        vi.mocked(getExperimentalStats).mockResolvedValue({
            effectiveness: 'unknown',
            ahi_trend_direction: null,
        })
    })

    afterEach(() => {
        // useIsMobile's ref is a module-scope singleton — reset the breakpoint
        // so mobile-branch tests never leak into later tests in this file.
        setMediaMatches(false)
    })

    async function mountView() {
        const wrapper = mount(AnalysisManagementView)
        await flushPromises()
        return wrapper
    }

    async function clickBatchDialogButton(
        label: string,
        dates: { from?: string; to?: string } = {},
    ): Promise<ReturnType<typeof mount>> {
        // The batch dialog teleports to document.body, so its buttons are
        // outside the wrapper; open it through the toolbar button first.
        const wrapper = mount(AnalysisManagementView, { attachTo: document.body })
        await flushPromises()
        const runBatch = wrapper.findAll('button').find((b) => b.text() === 'Run Batch')
        await runBatch!.trigger('click')
        await flushPromises()
        // The toolbar filter pickers render first; the dialog's From/To are the last two.
        const [fromPicker, toPicker] = wrapper.findAllComponents(DatePickerInput).slice(-2)
        if (dates.from) fromPicker.vm.$emit('update:modelValue', dates.from)
        if (dates.to) toPicker.vm.$emit('update:modelValue', dates.to)
        await flushPromises()
        const button = [...document.body.querySelectorAll('button')].find(
            (b) => b.textContent?.trim() === label,
        )
        expect(button, `dialog button "${label}"`).toBeDefined()
        button!.click()
        await flushPromises()
        wrapper.unmount()
        return wrapper
    }

    it.each([
        ['Analyze Missing', 'missing'],
        ['Re-analyze Outdated', 'stale'],
    ])('test_%s_without_dates_sends_only_%s_for_whole_profile', async (label, only) => {
        // Falsifiable: swap the two buttons' selectors, or send empty-string
        // dates when no picker is set; toMatchObject / not.toHaveProperty fail.
        await clickBatchDialogButton(label)

        expect(runBatchAnalysis).toHaveBeenCalledOnce()
        const [body] = vi.mocked(runBatchAnalysis).mock.calls[0]
        expect(body).toMatchObject({ only, store_results: true })
        expect(body).not.toHaveProperty('from_date')
        expect(body).not.toHaveProperty('to_date')
    })

    it('test_reanalyze_outdated_with_dates_sends_dates_and_only_stale', async () => {
        // Falsifiable: drop from_date/to_date from the only-button body (the
        // pickers silently ignored); toEqual fails.
        await clickBatchDialogButton('Re-analyze Outdated', {
            from: '2026-01-01',
            to: '2026-01-31',
        })

        expect(runBatchAnalysis).toHaveBeenCalledOnce()
        const [body] = vi.mocked(runBatchAnalysis).mock.calls[0]
        expect(body).toEqual({
            only: 'stale',
            from_date: '2026-01-01',
            to_date: '2026-01-31',
            modes: ['aasm'],
            store_results: true,
        })
    })

    it('test_analyze_missing_with_reversed_dates_shows_error_and_sends_nothing', async () => {
        const wrapper = await clickBatchDialogButton('Analyze Missing', {
            from: '2026-02-01',
            to: '2026-01-01',
        })

        expect(runBatchAnalysis).not.toHaveBeenCalled()
        expect((wrapper.vm as unknown as { error: string | null }).error).toBe(
            'From date must be before To date',
        )
    })

    it('test_date_range_run_sends_dates_without_only', async () => {
        // Guard: the date-range Run path must not gain an only filter.
        await clickBatchDialogButton('Run', { from: '2026-01-01', to: '2026-01-31' })

        expect(runBatchAnalysis).toHaveBeenCalledOnce()
        const [body] = vi.mocked(runBatchAnalysis).mock.calls[0]
        expect(body).not.toHaveProperty('only')
        expect(body).toMatchObject({
            from_date: '2026-01-01',
            to_date: '2026-01-31',
            store_results: true,
        })
    })

    it('test_handleCancelJob_409_silenced_and_list_still_refreshed', async () => {
        vi.mocked(cancelAnalysisJob).mockRejectedValueOnce(
            Object.assign(new Error('409 Conflict'), { status: 409 }),
        )
        const wrapper = await mountView()
        const fetchCallsBefore = vi.mocked(getAnalysisJobs).mock.calls.length
        const vm = wrapper.vm as unknown as {
            handleCancelJob: (jobId: string) => Promise<void>
            error: string | null
        }

        await vm.handleCancelJob('job-already-done')
        await flushPromises()

        expect(vm.error).toBeNull()
        expect(vi.mocked(getAnalysisJobs).mock.calls.length).toBeGreaterThan(fetchCallsBefore)
    })

    it('test_mobile_breakpoint_renders_card_list_not_table', async () => {
        // Falsifiable: with the old immutable matchMedia stub isMobile could
        // never become true, so the card branch was unreachable in unit tests.
        setMediaMatches(true)
        vi.mocked(getAnalysisSessions).mockResolvedValue({
            items: [
                {
                    session_id: 1470,
                    session_date: '2026-04-06',
                    has_analysis: true,
                    analysis_id: 7,
                    duration_hours: 9.6,
                },
            ],
            total: 1,
            limit: 25,
            offset: 0,
        })
        const wrapper = mount(AnalysisManagementView, {
            global: {
                // The card header links to session-analysis; no router is
                // installed in this suite, so resolve RouterLink to an anchor.
                components: { RouterLink: { props: ['to'], template: '<a><slot /></a>' } },
            },
        })
        await flushPromises()

        expect(wrapper.find('.card-list').exists()).toBe(true)
        expect(wrapper.find('table').exists()).toBe(false)
    })

    it('test_failed_assessment_load_leaves_the_session_list_working', async () => {
        vi.mocked(getExperimentalStats).mockRejectedValue(new Error('boom'))
        vi.mocked(getAnalysisSessions).mockResolvedValue({
            items: [
                {
                    session_id: 1470,
                    session_date: '2026-04-06',
                    has_analysis: true,
                    analysis_id: 7,
                    duration_hours: 9.6,
                },
            ],
            total: 1,
            limit: 25,
            offset: 0,
        })

        const wrapper = mount(AnalysisManagementView, {
            global: { components: { RouterLink: { template: '<a><slot /></a>' } } },
        })
        await flushPromises()

        expect(wrapper.find('.section-card .error-state').text()).toContain('boom')
        expect(wrapper.find('table').text()).toContain('2026-04-06')
    })
})
