import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'

vi.mock('vue-router')
vi.mock('@/api/stats', () => ({
    getSummary: vi.fn().mockRejectedValue(new Error('not under test')),
    getTrends: vi.fn().mockRejectedValue(new Error('not under test')),
}))
vi.mock('@/api/days', () => ({
    getDays: vi.fn().mockRejectedValue(new Error('not under test')),
}))
vi.mock('@/api/sessions', () => ({
    getSessions: vi.fn().mockRejectedValue(new Error('not under test')),
}))
vi.mock('@/api/health', () => ({ getHealthNights: vi.fn() }))

import { getHealthNights } from '@/api/health'
import DashboardView from '@/views/DashboardView.vue'

async function healthFromDate(): Promise<string | undefined> {
    const wrapper = mount(DashboardView, { global: { stubs: { RouterLink: true } } })
    await flushPromises()
    wrapper.unmount()
    return vi.mocked(getHealthNights).mock.calls[0]![0]!.from_date
}

describe('DashboardView health window', () => {
    // Node re-reads TZ on assignment, so each case sets a zone whose local
    // date differs from the UTC date at the pinned instant.
    beforeEach(() => {
        vi.useFakeTimers({ toFake: ['Date'] })
        vi.mocked(getHealthNights).mockResolvedValue({ items: [] } as never)
    })
    afterEach(() => {
        vi.useRealTimers()
        vi.unstubAllEnvs()
        vi.mocked(getHealthNights).mockReset()
    })

    it('test_early_morning_ahead_of_utc_uses_local_date', async () => {
        vi.stubEnv('TZ', 'Pacific/Auckland')
        // 05:00 NZDT on Oct 15 is still Oct 14 in UTC.
        vi.setSystemTime(new Date(2026, 9, 15, 5))

        expect(await healthFromDate()).toBe('2026-09-15')
    })

    it('test_evening_behind_utc_uses_local_date', async () => {
        vi.stubEnv('TZ', 'America/Los_Angeles')
        // 20:00 PDT on Oct 15 is already Oct 16 in UTC.
        vi.setSystemTime(new Date(2026, 9, 15, 20))

        expect(await healthFromDate()).toBe('2026-09-15')
    })
})
