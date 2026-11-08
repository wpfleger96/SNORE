import { apiGet, apiPost, apiDelete } from './client'
import type {
    PaginatedResponse,
    AnalysisListItem,
    AnalysisResult,
    AnalysisDeletePreview,
    AnalysisJobsListResponse,
} from '@/types'

export interface AnalysisSessionsParams {
    limit?: number
    offset?: number
    from_date?: string
    to_date?: string
    analyzed_only?: boolean
    sort_by?: string
}

export const getAnalysisSessions = apiGet<
    PaginatedResponse<AnalysisListItem>,
    [params?: AnalysisSessionsParams]
>('/analysis/sessions', (params = {}) => ({ params }))

export const getAnalysis = apiGet<AnalysisResult, [sessionId: number]>(
    (sessionId) => `/sessions/${sessionId}/analysis`,
)

export const runAnalysis = apiPost<
    AnalysisResult,
    [sessionId: number, body?: { modes?: string[]; store_results?: boolean }]
>(
    (sessionId) => `/sessions/${sessionId}/analysis`,
    (_sessionId, body = {}) => ({ data: { modes: ['aasm'], store_results: true, ...body } }),
)

export const deleteAnalysis = apiDelete<
    { deleted_count: number },
    [body: { session_ids: number[]; all_versions?: boolean }]
>('/analysis', (body) => ({ data: body }))

export const getAnalysisDeletePreview = apiGet<
    AnalysisDeletePreview,
    [params?: { session_ids?: number[]; all_versions?: boolean }]
>('/analysis/delete-preview', (params = {}) => ({ params }))

export const runBatchAnalysis = apiPost<
    { job_id: string; session_count: number },
    [
        body: {
            from_date?: string
            to_date?: string
            only?: 'missing' | 'stale'
            modes?: string[]
            store_results?: boolean
        },
    ]
>('/analysis/batch', (body) => ({ data: body }))

export const getAnalysisJobs = apiGet<AnalysisJobsListResponse>('/analysis/jobs')

export const cancelAnalysisJob = apiDelete<void, [jobId: string]>(
    (jobId) => `/analysis/jobs/${jobId}`,
)
