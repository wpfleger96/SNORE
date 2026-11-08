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
function mountPicker(
    modes: AnalysisMode[],
    primaryMode: AnalysisMode | null = null,
    props: { showStoredRera?: boolean } = {},
) {
    const state = reactive({ modes, primaryMode })
    const wrapper = mount({
        render: () =>
            h(AnalysisModePicker, {
                ...props,
                modes: state.modes,
                primaryMode: state.primaryMode,
                'onUpdate:modes': (v: AnalysisMode[]) => (state.modes = v),
                'onUpdate:primaryMode': (v: AnalysisMode | null) => (state.primaryMode = v),
            }),
    })
    return { wrapper, state, root: wrapper.element as HTMLElement }
}

function triggerText(root: HTMLElement) {
    return advancedTrigger(root)!.textContent!.trim()
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
        expect(triggerText(root)).toBe('Advanced: RERA flags from AASM')
        expect(trigger.getAttribute('aria-expanded')).toBe('false')
        expect(pickerGroup(root, RERA_GROUP)).toBeNull()
    })

    it('test_hidden_stored_rera_choice_omits_advanced', () => {
        const { root } = mountPicker(['aasm', 'resmed'], 'resmed', { showStoredRera: false })

        expect(advancedTrigger(root)).toBeNull()
        expect(pickerGroup(root, RERA_GROUP)).toBeNull()
    })

    it('test_explicit_pick_from_parent_names_the_trigger', () => {
        const { root } = mountPicker(['aasm', 'resmed'], 'resmed')

        expect(triggerText(root)).toBe('Advanced: RERA flags from ResMed')
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
        expect(state.primaryMode).toBeNull()
        expect(triggerText(root)).toBe('Advanced: RERA flags from AASM')
    })

    it('test_default_primary_without_aasm_is_first_selected_mode', async () => {
        const { root } = mountPicker(['resmed'])

        await clickMode(root, 'Modes', 'AASM Relaxed')

        expect(triggerText(root)).toBe('Advanced: RERA flags from AASM Relaxed')
    })

    it('test_explicit_pick_updates_trigger_and_survives_adding_a_mode', async () => {
        const { state, root } = mountPicker(['aasm_relaxed', 'resmed'])

        await clickMode(root, RERA_GROUP, 'ResMed')
        await clickMode(root, 'Modes', 'AASM')

        expect(state.modes).toEqual(['aasm', 'aasm_relaxed', 'resmed'])
        expect(state.primaryMode).toBe('resmed')
        expect(triggerText(root)).toBe('Advanced: RERA flags from ResMed')
    })

    it('test_reclicking_the_pressed_default_pins_it', async () => {
        const { state, root } = mountPicker(['aasm_relaxed', 'resmed'])

        await clickMode(root, RERA_GROUP, 'AASM Relaxed')
        expect(state.primaryMode).toBe('aasm_relaxed')
        expect(pressedModes(root, RERA_GROUP)).toEqual(['AASM Relaxed'])

        // Pinned, so adding AASM (the default when selected) does not take over.
        await clickMode(root, 'Modes', 'AASM')
        expect(triggerText(root)).toBe('Advanced: RERA flags from AASM Relaxed')
    })

    it('test_deselecting_explicit_pick_falls_back_to_default', async () => {
        const { state, root } = mountPicker(['aasm', 'aasm_relaxed', 'resmed'])

        await clickMode(root, RERA_GROUP, 'ResMed')
        await clickMode(root, 'Modes', 'ResMed')
        expect(state.primaryMode).toBeNull()
        expect(triggerText(root)).toBe('Advanced: RERA flags from AASM')

        // The override is gone: re-adding ResMed does not restore it.
        await clickMode(root, 'Modes', 'ResMed')
        expect(triggerText(root)).toBe('Advanced: RERA flags from AASM')
    })

    it('test_empty_selection_asks_for_a_mode', async () => {
        const { wrapper, state, root } = mountPicker(['aasm'])

        await clickMode(root, 'Modes', 'AASM')

        expect(state.modes).toEqual([])
        expect(wrapper.text()).toContain('Select at least one mode.')
    })
})
