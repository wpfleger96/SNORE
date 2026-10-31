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

    it('test_handleAnalyzeMissing_sends_missing_only_without_dates', async () => {
        // Falsifiable: remove missing_only from the payload in handleAnalyzeMissing;
        // objectContaining({ missing_only: true }) fails.
        const wrapper = await mountView()
        const vm = wrapper.vm as unknown as { handleAnalyzeMissing: () => Promise<void> }

        await vm.handleAnalyzeMissing()
        await flushPromises()

        expect(runBatchAnalysis).toHaveBeenCalledOnce()
        const [body] = vi.mocked(runBatchAnalysis).mock.calls[0]
        expect(body).toMatchObject({ missing_only: true, store_results: true })
        expect(body).not.toHaveProperty('from_date')
        expect(body).not.toHaveProperty('to_date')
    })

    it('test_handleBatchRun_does_not_include_missing_only', async () => {
        // Guard: the existing batch-run path must not gain missing_only.
        // batchFrom/batchTo start as empty strings; handleBatchRun proceeds without
        // the date-validation check (both falsy) and fires runBatchAnalysis.
        const wrapper = await mountView()
        const vm = wrapper.vm as unknown as { handleBatchRun: () => Promise<void> }

        await vm.handleBatchRun()
        await flushPromises()

        expect(runBatchAnalysis).toHaveBeenCalledOnce()
        const [body] = vi.mocked(runBatchAnalysis).mock.calls[0]
        expect(body).not.toHaveProperty('missing_only')
        expect(body).toMatchObject({ store_results: true })
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

    it('test_therapy_assessment_shows_effectiveness_and_trend_under_a_banner', async () => {
        vi.mocked(getExperimentalStats).mockResolvedValue({
            effectiveness: 'good',
            ahi_trend_direction: 'worsening',
        })

        const wrapper = await mountView()

        const card = wrapper.find('.section-card')
        expect(card.findAll('[role="note"]')).toHaveLength(1)
        expect(card.text()).toContain('good')
        const trend = card.find('.trend-badge')
        expect(trend.text()).toBe('worsening')
        expect(trend.classes()).toContain('trend-worsening')
    })

    it('test_therapy_assessment_hides_unknown_band_but_keeps_a_real_trend', async () => {
        vi.mocked(getExperimentalStats).mockResolvedValue({
            effectiveness: 'unknown',
            ahi_trend_direction: 'stable',
        })

        const wrapper = await mountView()

        const card = wrapper.find('.section-card')
        expect(card.text()).not.toContain('unknown')
        expect(card.text()).not.toContain('Effectiveness')
        expect(card.find('.trend-badge').text()).toBe('stable')
    })

    it('test_therapy_assessment_shows_empty_state_when_nothing_to_show', async () => {
        const wrapper = await mountView()

        const card = wrapper.find('.section-card')
        expect(card.find('.assessment-row').exists()).toBe(false)
        expect(card.find('.assessment-empty').exists()).toBe(true)
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
