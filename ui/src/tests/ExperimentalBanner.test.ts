import { describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { ref } from 'vue'
import { makeAuthMock } from './helpers/mockUseAuth'

vi.mock('@/composables/useAuth', () => ({ useAuth: () => makeAuthMock() }))
vi.mock('@/composables/useAvailableDates', () => ({
    useAvailableDates: () => ({
        load: vi.fn(),
        isDateDisabled: () => false,
        minValue: ref(undefined),
        maxValue: ref(undefined),
    }),
}))
vi.mock('@/composables/useValidationRuns', () => ({
    useValidationRuns: () => ({
        runs: ref([]),
        runsForType: () => [],
        isActive: () => false,
        refresh: vi.fn().mockResolvedValue(undefined),
        getDetail: vi.fn(),
        enqueue: vi.fn(),
        remove: vi.fn(),
    }),
}))
vi.mock('@/components/DatePickerInput.vue', () => ({ default: { template: '<input />' } }))

import ExperimentalBanner from '@/components/ExperimentalBanner.vue'
import ValidationPanelShell from '@/components/validation/ValidationPanelShell.vue'

const DEFAULT_BODY = 'These are internally-consistent trend instruments'

describe('ExperimentalBanner', () => {
    it('test_default_title_and_body', () => {
        const wrapper = mount(ExperimentalBanner)

        expect(wrapper.find('.font-medium').text()).toBe('Experimental metric.')
        expect(wrapper.text()).toContain(DEFAULT_BODY)
        expect(wrapper.find('svg').exists()).toBe(true)
    })

    it('test_props_override_title_and_body', () => {
        const wrapper = mount(ExperimentalBanner, {
            props: { title: 'Heads up.', body: 'Custom note.' },
        })

        expect(wrapper.text()).toBe('Heads up. Custom note.')
    })
})

describe('ValidationPanelShell experimental banner', () => {
    const baseProps = { validatorType: 'fl' as const, filenameBase: 'fl-validation' }

    it('test_experimental_shell_renders_default_banner', () => {
        const wrapper = mount(ValidationPanelShell, { props: { ...baseProps, experimental: true } })

        const banner = wrapper.findComponent(ExperimentalBanner)
        expect(banner.exists()).toBe(true)
        expect(banner.text()).toContain('Experimental metric.')
        expect(banner.text()).toContain(DEFAULT_BODY)
    })

    it('test_experimental_note_replaces_banner_body', () => {
        const wrapper = mount(ValidationPanelShell, {
            props: { ...baseProps, experimental: true, experimentalNote: 'Panel note.' },
        })

        expect(wrapper.findComponent(ExperimentalBanner).text()).toBe(
            'Experimental metric. Panel note.',
        )
    })

    it('test_non_experimental_shell_has_no_banner', () => {
        const wrapper = mount(ValidationPanelShell, { props: baseProps })

        expect(wrapper.findComponent(ExperimentalBanner).exists()).toBe(false)
    })
})
