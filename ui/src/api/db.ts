import { apiGet, apiPost } from './client'
import type { DatabaseStatsPublic, ResetJobEnqueued, ResetJobStatus, VacuumResult } from '@/types'

// A hung request must count as a failed poll, or it would stall the poll loop.
const RESET_JOB_STATUS_TIMEOUT_MS = 15_000

export interface ResetRequest {
    include_accounts?: boolean
}

export const getDbStats = apiGet<DatabaseStatsPublic>('/db/stats')
export const vacuumDb = apiPost<VacuumResult>('/db/vacuum')
export const resetDb = apiPost<ResetJobEnqueued, [body?: ResetRequest]>('/db/reset', (body) =>
    body ? { data: body } : {},
)
// The token is a bearer secret. It stays in a header because URLs land in proxy logs.
export const getResetJobStatus = apiGet<ResetJobStatus, [token: string]>(
    '/db/reset-jobs/status',
    (token) => ({
        headers: { 'X-Reset-Job-Token': token },
        timeout: RESET_JOB_STATUS_TIMEOUT_MS,
    }),
)
