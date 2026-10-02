import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import type { RecordsData } from '@/types'
import RecordsPanel from '@/components/RecordsPanel.vue'

const ENTRY = { best: [['2026-09-01', 1.2]], worst: [['2026-09-02', 6.3]] }

function markTiers(records: RecordsData): Record<string, string[]> {
    const wrapper = mount(RecordsPanel, { props: { records, loading: false } })
    return Object.fromEntries(
        wrapper
            .findAll('.record-card h4')
            .map((h) => [
                h.text(),
                h.findAll('.provenance-mark').map((m) => m.attributes('aria-label')!.split(':')[0]),
            ]),
    )
}

describe('RecordsPanel provenance', () => {
    it('test_every_record_metric_is_marked_derived', () => {
        // Records rank per-day aggregates, so every metric is Derived — including
        // AHI, whose tier differs by schema (RecordsResponse.ahi is Derived).
        const records = {
            ahi: ENTRY,
            leak: ENTRY,
            therapy_hours: ENTRY,
            spo2_min: ENTRY,
            total_sleep_hours: ENTRY,
        } as unknown as RecordsData

        expect(markTiers(records)).toEqual({
            AHI: ['Derived'],
            'Leak (L/min)': ['Derived'],
            'Therapy Hours': ['Derived'],
            'SpO₂ Min (%)': ['Derived'],
            'Total Sleep (hrs)': ['Derived'],
        })
    })
})
