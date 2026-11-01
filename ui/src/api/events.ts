import { apiGet } from './client'
import type { EventItem } from '@/types'

export const getSessionEvents = apiGet<EventItem[], [sessionId: number, eventType?: string]>(
    (sessionId) => `/sessions/${sessionId}/events`,
    (_sessionId, eventType) => ({ params: eventType ? { event_type: eventType } : {} }),
)
