import { describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'

vi.mock('@/components/InfoHint.vue', () => ({
    default: {
        props: ['glossaryKey', 'provenance'],
        template: '<span class="info-hint-stub" :data-provenance="provenance" />',
    },
}))

vi.mock('@lucide/vue', () => ({
    Sigma: { template: '<svg class="icon-sigma-stub" />' },
    FlaskConical: { template: '<svg class="icon-flask-stub" />' },
}))

import StatCard from '@/components/StatCard.vue'

function mountCard(props: Record<string, unknown>) {
    return mount(StatCard, { props: { label: 'Metric', value: 1, ...props } })
}

function markLabels(wrapper: ReturnType<typeof mountCard>): string[] {
    return wrapper.findAll('.stat-label .provenance-mark').map((m) => m.attributes('aria-label')!)
}

describe('StatCard provenance mark', () => {
    it('test_derived_field_renders_sigma_mark', () => {
        const wrapper = mountCard({ field: 'leak_mean' })

        expect(markLabels(wrapper)).toHaveLength(1)
        expect(markLabels(wrapper)[0]).toMatch(/^Derived/)
        expect(wrapper.find('.icon-sigma-stub').exists()).toBe(true)
    })

    it('test_experimental_field_renders_flask_mark', () => {
        const wrapper = mountCard({ field: 'rera_index' })

        expect(markLabels(wrapper)[0]).toMatch(/^Experimental/)
        expect(wrapper.find('.icon-flask-stub').exists()).toBe(true)
    })

    it('test_device_field_renders_no_mark', () => {
        const wrapper = mountCard({ field: 'obstructive_apneas' })

        expect(markLabels(wrapper)).toEqual([])
        // Nothing to show next to the label at all without a glossary key.
        expect(wrapper.find('.stat-hints').exists()).toBe(false)
    })

    it('test_no_field_renders_no_mark', () => {
        expect(markLabels(mountCard({}))).toEqual([])
    })

    it('test_schema_selects_tier_for_schema_dependent_field', () => {
        expect(markLabels(mountCard({ field: 'ahi', schema: 'DayDetail' }))).toEqual([])
        expect(markLabels(mountCard({ field: 'ahi', schema: 'SessionStatistics' }))[0]).toMatch(
            /^Derived/,
        )
    })

    it('test_source_overrides_schema_tier', () => {
        const wrapper = mountCard({ field: 'ahi', schema: 'DayDetail', source: 'derived' })

        expect(markLabels(wrapper)[0]).toMatch(/^Derived/)
    })

    it('test_provenance_prop_overrides_field_lookup', () => {
        const wrapper = mountCard({ field: 'leak_mean', provenance: 'experimental' })

        expect(markLabels(wrapper)).toHaveLength(1)
        expect(markLabels(wrapper)[0]).toMatch(/^Experimental/)
    })

    it('test_info_hint_receives_resolved_tier', () => {
        const wrapper = mountCard({ field: 'leak_mean', glossaryKey: 'leak' })

        expect(wrapper.find('.info-hint-stub').attributes('data-provenance')).toBe('derived')
    })
})
