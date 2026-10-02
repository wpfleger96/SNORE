import { describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'

vi.mock('@/components/InfoHint.vue', () => ({
    default: {
        props: ['glossaryKey'],
        template: '<span class="info-hint-stub" />',
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
        const wrapper = mountCard({ field: 'SessionStatistics.leak_mean' })

        expect(markLabels(wrapper)).toHaveLength(1)
        expect(markLabels(wrapper)[0]).toMatch(/^Derived/)
        expect(wrapper.find('.icon-sigma-stub').exists()).toBe(true)
    })

    it('test_experimental_field_renders_flask_mark', () => {
        const wrapper = mountCard({ field: 'DayDetail.rera_index' })

        expect(markLabels(wrapper)[0]).toMatch(/^Experimental/)
        expect(wrapper.find('.icon-flask-stub').exists()).toBe(true)
    })

    it('test_device_field_renders_no_mark', () => {
        const wrapper = mountCard({ field: 'SessionStatistics.obstructive_apneas' })

        expect(markLabels(wrapper)).toEqual([])
        // Nothing to show next to the label at all without a glossary key.
        expect(wrapper.find('.stat-hints').exists()).toBe(false)
    })

    it('test_no_field_renders_no_mark', () => {
        expect(markLabels(mountCard({}))).toEqual([])
    })

    it('test_provenance_prop_overrides_field_lookup', () => {
        const wrapper = mountCard({
            field: 'SessionStatistics.leak_mean',
            provenance: 'experimental',
        })

        expect(markLabels(wrapper)).toHaveLength(1)
        expect(markLabels(wrapper)[0]).toMatch(/^Experimental/)
    })

    it('test_glossary_key_without_mark_shows_only_info_hint', () => {
        const wrapper = mountCard({
            field: 'SessionStatistics.obstructive_apneas',
            glossaryKey: 'obstructive_apneas',
        })

        expect(markLabels(wrapper)).toEqual([])
        expect(wrapper.find('.stat-hints .info-hint-stub').exists()).toBe(true)
    })
})
