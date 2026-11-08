import { describe, expect, it } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { h, reactive } from 'vue'
import AnalysisModePicker from '@/components/AnalysisModePicker.vue'
import type { AnalysisMode } from '@/api/analysis'
import {
    advancedTrigger,
    clickMode,
    pickerGroup,
    pressedModes,
    RERA_GROUP,
} from './helpers/modePicker'

// The host owns the v-model state, as the analysis views do.
function mountPicker(modes: AnalysisMode[], primaryMode: AnalysisMode = modes[0]!) {
    const state = reactive({ modes, primaryMode })
    const wrapper = mount({
        render: () =>
            h(AnalysisModePicker, {
                modes: state.modes,
                primaryMode: state.primaryMode,
                'onUpdate:modes': (v: AnalysisMode[]) => (state.modes = v),
                'onUpdate:primaryMode': (v: AnalysisMode) => (state.primaryMode = v),
            }),
    })
    return { wrapper, state, root: wrapper.element as HTMLElement }
}

describe('AnalysisModePicker', () => {
    it('test_every_detection_mode_is_offered', () => {
        const { root } = mountPicker(['aasm'])

        const labels = [...pickerGroup(root, 'Modes')!.querySelectorAll('button')].map((b) =>
            b.textContent!.trim(),
        )
        expect(labels).toEqual(['AASM', 'AASM Relaxed', 'ResMed'])
    })

    it('test_single_mode_hides_advanced_disclosure', () => {
        const { root } = mountPicker(['resmed'])

        expect(advancedTrigger(root)).toBeNull()
        expect(pickerGroup(root, RERA_GROUP)).toBeNull()
    })

    it('test_several_modes_show_advanced_collapsed_naming_current_choice', () => {
        const { root } = mountPicker(['aasm', 'resmed'])

        const trigger = advancedTrigger(root)!
        expect(trigger.textContent!.trim()).toBe('Advanced: RERA flags from AASM')
        expect(trigger.getAttribute('aria-expanded')).toBe('false')
        expect(pickerGroup(root, RERA_GROUP)).toBeNull()
    })

    it('test_selected_non_default_primary_passed_on_mount_is_kept', () => {
        const { root, state } = mountPicker(['aasm', 'resmed'], 'resmed')

        expect(state.primaryMode).toBe('resmed')
        expect(advancedTrigger(root)!.textContent!.trim()).toBe('Advanced: RERA flags from ResMed')
    })

    it('test_expanding_advanced_offers_only_selected_modes', async () => {
        const { root } = mountPicker(['aasm', 'resmed'])

        advancedTrigger(root)!.click()
        await flushPromises()

        const options = [...pickerGroup(root, RERA_GROUP)!.querySelectorAll('button')].map((b) =>
            b.textContent!.trim(),
        )
        expect(options).toEqual(['AASM', 'ResMed'])
        expect(pressedModes(root, RERA_GROUP)).toEqual(['AASM'])
    })

    it('test_adding_aasm_makes_it_the_default_primary', async () => {
        const { state, root } = mountPicker(['resmed'])

        await clickMode(root, 'Modes', 'AASM')

        expect(state.modes).toEqual(['aasm', 'resmed'])
        expect(state.primaryMode).toBe('aasm')
    })

    it('test_default_primary_without_aasm_is_first_selected_mode', async () => {
        const { state, root } = mountPicker(['resmed'])

        await clickMode(root, 'Modes', 'AASM Relaxed')

        expect(state.primaryMode).toBe('aasm_relaxed')
        expect(advancedTrigger(root)!.textContent!.trim()).toBe(
            'Advanced: RERA flags from AASM Relaxed',
        )
    })

    it('test_explicit_pick_updates_trigger_and_survives_adding_a_mode', async () => {
        const { state, root } = mountPicker(['aasm_relaxed', 'resmed'])

        await clickMode(root, RERA_GROUP, 'ResMed')
        await clickMode(root, 'Modes', 'AASM')

        expect(state.modes).toEqual(['aasm', 'aasm_relaxed', 'resmed'])
        expect(state.primaryMode).toBe('resmed')
        expect(advancedTrigger(root)!.textContent!.trim()).toBe('Advanced: RERA flags from ResMed')
    })

    it('test_deselecting_explicit_pick_falls_back_to_default', async () => {
        const { state, root } = mountPicker(['aasm', 'aasm_relaxed', 'resmed'])

        await clickMode(root, RERA_GROUP, 'ResMed')
        await clickMode(root, 'Modes', 'ResMed')
        expect(state.primaryMode).toBe('aasm')

        // The override is gone: re-adding ResMed does not restore it.
        await clickMode(root, 'Modes', 'ResMed')
        expect(state.primaryMode).toBe('aasm')
    })

    it('test_initial_primary_outside_selection_is_corrected', () => {
        const { state } = mountPicker(['resmed'], 'aasm')

        expect(state.primaryMode).toBe('resmed')
    })

    it('test_empty_selection_asks_for_a_mode', async () => {
        const { wrapper, state, root } = mountPicker(['aasm'])

        await clickMode(root, 'Modes', 'AASM')

        expect(state.modes).toEqual([])
        expect(wrapper.text()).toContain('Select at least one mode.')
    })
})
