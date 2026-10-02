import { afterEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'

// Stub the Popover family so tests run without DOM teleport / floating-ui APIs.
// The stubs model the controlled `open` state: the root provides its instance,
// the trigger toggles it on click, and the content renders only while open.
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
    PopoverHeader: { template: '<div><slot /></div>' },
    PopoverTitle: { template: '<span class="popover-title-stub"><slot /></span>' },
    PopoverDescription: { template: '<p class="popover-description-stub"><slot /></p>' },
}))

// Stub the lucide icon to avoid SVG rendering issues.
vi.mock('@lucide/vue', () => ({
    Info: { template: '<svg class="icon-info-stub" />' },
}))

import InfoHint from '@/components/InfoHint.vue'

afterEach(() => {
    vi.useRealTimers()
})

describe('InfoHint content resolution', () => {
    it('test_valid_glossary_key_resolves_label_in_aria', () => {
        const wrapper = mount(InfoHint, { props: { glossaryKey: 'ahi' } })

        const trigger = wrapper.find('button[type="button"]')
        expect(trigger.attributes('aria-label')).toContain('AHI')
    })

    it('test_valid_glossary_key_renders_short_text', async () => {
        const wrapper = mount(InfoHint, { props: { glossaryKey: 'ahi' } })

        await wrapper
            .find('button[type="button"]')
            .trigger('pointerenter', { pointerType: 'mouse' })

        expect(wrapper.find('.popover-description-stub').text()).toContain('Apnea-Hypopnea Index')
    })

    it('test_unknown_glossary_key_does_not_throw_and_renders_empty_label', () => {
        // Should not throw; aria-label should end with empty resolved label.
        let wrapper: ReturnType<typeof mount> | undefined
        expect(() => {
            wrapper = mount(InfoHint, { props: { glossaryKey: 'nonexistent_key_xyz' } })
        }).not.toThrow()

        const trigger = wrapper!.find('button[type="button"]')
        expect(trigger.attributes('aria-label')).toBe('More information about ')
    })

    it('test_explicit_label_overrides_glossary_entry', () => {
        const wrapper = mount(InfoHint, {
            props: { glossaryKey: 'ahi', label: 'My Custom Label' },
        })

        const trigger = wrapper.find('button[type="button"]')
        expect(trigger.attributes('aria-label')).toContain('My Custom Label')
        expect(trigger.attributes('aria-label')).not.toContain('AHI')
    })

    it('test_explicit_short_overrides_glossary_entry', async () => {
        const wrapper = mount(InfoHint, {
            props: { glossaryKey: 'ahi', short: 'Custom short text.' },
        })

        await wrapper
            .find('button[type="button"]')
            .trigger('pointerenter', { pointerType: 'mouse' })

        expect(wrapper.find('.popover-description-stub').text()).toBe('Custom short text.')
    })
})

describe('InfoHint hover behavior', () => {
    it('test_mouse_pointerenter_opens_content', async () => {
        const wrapper = mount(InfoHint, { props: { glossaryKey: 'ahi' } })
        expect(wrapper.find('.popover-content-stub').exists()).toBe(false)

        await wrapper
            .find('button[type="button"]')
            .trigger('pointerenter', { pointerType: 'mouse' })

        expect(wrapper.find('.popover-content-stub').exists()).toBe(true)
    })

    it('test_touch_pointerenter_does_not_open_content', async () => {
        const wrapper = mount(InfoHint, { props: { glossaryKey: 'ahi' } })

        await wrapper
            .find('button[type="button"]')
            .trigger('pointerenter', { pointerType: 'touch' })

        expect(wrapper.find('.popover-content-stub').exists()).toBe(false)
    })

    it('test_mouse_pointerleave_closes_content_after_delay', async () => {
        vi.useFakeTimers()
        const wrapper = mount(InfoHint, { props: { glossaryKey: 'ahi' } })
        const button = wrapper.find('button[type="button"]')

        await button.trigger('pointerenter', { pointerType: 'mouse' })
        expect(wrapper.find('.popover-content-stub').exists()).toBe(true)

        await button.trigger('pointerleave', { pointerType: 'mouse' })
        // Still open before the grace period elapses.
        expect(wrapper.find('.popover-content-stub').exists()).toBe(true)

        vi.advanceTimersByTime(150)
        await wrapper.vm.$nextTick()
        expect(wrapper.find('.popover-content-stub').exists()).toBe(false)
    })

    it('test_entering_content_within_grace_period_keeps_open', async () => {
        vi.useFakeTimers()
        const wrapper = mount(InfoHint, { props: { glossaryKey: 'ahi' } })
        const button = wrapper.find('button[type="button"]')

        await button.trigger('pointerenter', { pointerType: 'mouse' })
        await button.trigger('pointerleave', { pointerType: 'mouse' })

        // Pointer reaches the content before the pending close fires.
        await wrapper
            .find('.popover-content-stub')
            .trigger('pointerenter', { pointerType: 'mouse' })

        vi.advanceTimersByTime(150)
        await wrapper.vm.$nextTick()
        expect(wrapper.find('.popover-content-stub').exists()).toBe(true)
    })

    it('test_click_toggles_open_state', async () => {
        const wrapper = mount(InfoHint, { props: { glossaryKey: 'ahi' } })

        await wrapper.find('.popover-trigger-stub').trigger('click')
        expect(wrapper.find('.popover-content-stub').exists()).toBe(true)

        await wrapper.find('.popover-trigger-stub').trigger('click')
        expect(wrapper.find('.popover-content-stub').exists()).toBe(false)
    })
})
