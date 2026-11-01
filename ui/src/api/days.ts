import { apiGet } from './client'
import type {
    PaginatedResponse,
    DayListItem,
    DayDetail,
    DayExperimentalMetrics,
    DateListResponse,
} from '@/types'

export interface DaysParams {
    limit?: number
    offset?: number
    from_date?: string
    to_date?: string
    device_id?: number
}

export const getDays = apiGet<PaginatedResponse<DayListItem>, [params?: DaysParams]>(
    '/days/',
    (params = {}) => ({ params }),
)

export const getDay = apiGet<DayDetail, [date: string]>((date) => `/days/${date}`)

export const getDayExperimental = apiGet<DayExperimentalMetrics, [date: string]>(
    (date) => `/days/${date}/experimental`,
)

export const getDates = apiGet<DateListResponse>('/days/dates')
