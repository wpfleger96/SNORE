import { describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'

// Stub the Popover family (same controlled-open model as InfoHint.test.ts).
vi.mock('@/components/ui/popover', () => ({
    Popover: {
        props: ['open'],
        emits: ['update:open'],
        provide() {
            return { popoverRoot: this }
        },
        template: '<div class="popover-stub"><slot /></div>',
    },
    PopoverTrigger: {
        props: ['asChild'],
        inject: ['popoverRoot'],
        template:
            '<div class="popover-trigger-stub" @click="popoverRoot.$emit(\'update:open\', !popoverRoot.open)"><slot /></div>',
    },
    PopoverContent: {
        inject: ['popoverRoot'],
        template: '<div v-if="popoverRoot.open" class="popover-content-stub"><slot /></div>',
    },
}))

vi.mock('@lucide/vue', () => ({
    Sigma: { template: '<svg class="icon-sigma-stub" />' },
    FlaskConical: { template: '<svg class="icon-flask-stub" />' },
}))

import ProvenanceMark from '@/components/ProvenanceMark.vue'
import ProvenanceLegend from '@/components/ProvenanceLegend.vue'

describe('ProvenanceMark', () => {
    it('test_device_renders_nothing', () => {
        const wrapper = mount(ProvenanceMark, { props: { provenance: 'device' } })

        expect(wrapper.find('button').exists()).toBe(false)
        expect(wrapper.text()).toBe('')
    })

    it('test_derived_icon_has_sigma_and_aria_label', () => {
        const wrapper = mount(ProvenanceMark, { props: { provenance: 'derived' } })

        const button = wrapper.find('button')
        expect(button.attributes('aria-label')).toBe('Derived: Computed by SNORE from device data.')
        expect(button.find('.icon-sigma-stub').exists()).toBe(true)
        expect(button.text()).toBe('')
    })

    it('test_experimental_icon_has_flask_and_aria_label', () => {
        const wrapper = mount(ProvenanceMark, { props: { provenance: 'experimental' } })

        const button = wrapper.find('button')
        expect(button.attributes('aria-label')).toMatch(/^Experimental: SNORE's own heuristic/)
        expect(button.find('.icon-flask-stub').exists()).toBe(true)
    })

    it('test_click_opens_popover_with_tier_and_note', async () => {
        const wrapper = mount(ProvenanceMark, { props: { provenance: 'derived' } })
        expect(wrapper.find('.popover-content-stub').exists()).toBe(false)

        await wrapper.find('.popover-trigger-stub').trigger('click')

        const content = wrapper.find('.popover-content-stub')
        expect(content.text()).toBe('Derived — Computed by SNORE from device data.')
        expect(content.find('.icon-sigma-stub').exists()).toBe(true)
    })

    it('test_mouse_hover_opens_popover', async () => {
        const wrapper = mount(ProvenanceMark, { props: { provenance: 'experimental' } })

        await wrapper.find('button').trigger('pointerenter', { pointerType: 'mouse' })

        expect(wrapper.find('.popover-content-stub').text()).toMatch(/^Experimental — /)
    })
})

describe('ProvenanceLegend', () => {
    it('test_lists_marked_tiers_and_unmarked_note', () => {
        const wrapper = mount(ProvenanceLegend)

        const terms = wrapper.findAll('li .font-medium').map((el) => el.text())
        expect(terms).toEqual(['Derived', 'Experimental'])
        expect(wrapper.text()).toContain('Computed by SNORE from device data.')
        expect(wrapper.text()).toContain('Unmarked values are device-reported.')
    })
})
