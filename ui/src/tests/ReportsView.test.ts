import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'

vi.mock('@/api/reports', () => ({ getSummaryReport: vi.fn(), getComparisonReport: vi.fn() }))
vi.mock('@/api/export', () => ({ downloadBlob: vi.fn() }))
vi.mock('@/api/rx', () => ({ getRxHistory: vi.fn().mockResolvedValue([]) }))

vi.mock('@/composables/useAvailableDates', () => ({
    useAvailableDates: () => ({
        load: vi.fn(),
        isDateDisabled: () => false,
        minValue: undefined,
        maxValue: undefined,
    }),
}))

vi.mock('@/components/DatePickerInput.vue', () => ({
    default: { props: ['modelValue'], template: '<input :value="modelValue" />' },
}))

import ReportsView from '@/views/ReportsView.vue'

async function rangeAfterPreset(label: string): Promise<string[]> {
    const wrapper: VueWrapper = mount(ReportsView)
    await flushPromises()
    await wrapper
        .findAll('button')
        .find((b) => b.text().trim() === label)!
        .trigger('click')
    const values = wrapper.findAll('input').map((i) => (i.element as HTMLInputElement).value)
    wrapper.unmount()
    return values
}

describe('ReportsView date presets', () => {
    // Node re-reads TZ on assignment, so each case sets a zone whose local
    // date differs from the UTC date at the pinned instant.
    beforeEach(() => vi.useFakeTimers({ toFake: ['Date'] }))
    afterEach(() => {
        vi.useRealTimers()
        vi.unstubAllEnvs()
    })

    it('test_preset_early_morning_ahead_of_utc_uses_local_dates', async () => {
        vi.stubEnv('TZ', 'Pacific/Auckland')
        // 05:00 NZDT on Oct 15 is still Oct 14 in UTC.
        vi.setSystemTime(new Date(2026, 9, 15, 5))

        expect(await rangeAfterPreset('30d')).toEqual(['2026-09-15', '2026-10-15'])
    })

    it('test_preset_evening_behind_utc_uses_local_dates', async () => {
        vi.stubEnv('TZ', 'America/Los_Angeles')
        // 20:00 PDT on Oct 15 is already Oct 16 in UTC.
        vi.setSystemTime(new Date(2026, 9, 15, 20))

        expect(await rangeAfterPreset('30d')).toEqual(['2026-09-15', '2026-10-15'])
    })
})
