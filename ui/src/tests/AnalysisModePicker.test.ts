import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import { h, reactive } from 'vue'
import AnalysisModePicker from '@/components/AnalysisModePicker.vue'
import type { AnalysisMode } from '@/api/analysis'
import { clickMode, pickerGroup, pressedModes } from './helpers/modePicker'

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

    it('test_single_mode_hides_primary_picker', () => {
        const { root } = mountPicker(['resmed'])

        expect(pickerGroup(root, 'Primary mode')).toBeNull()
    })

    it('test_adding_a_mode_shows_primary_picker_on_the_first_selected_mode', async () => {
        const { state, root } = mountPicker(['resmed'])

        await clickMode(root, 'Modes', 'AASM')

        expect(state.modes).toEqual(['aasm', 'resmed'])
        expect(state.primaryMode).toBe('resmed')
        expect(pressedModes(root, 'Primary mode')).toEqual(['ResMed'])
    })

    it('test_primary_picker_offers_only_selected_modes', async () => {
        const { state, root } = mountPicker(['aasm', 'resmed'])

        await clickMode(root, 'Primary mode', 'ResMed')

        const options = [...pickerGroup(root, 'Primary mode')!.querySelectorAll('button')].map(
            (b) => b.textContent!.trim(),
        )
        expect(options).toEqual(['AASM', 'ResMed'])
        expect(state.primaryMode).toBe('resmed')
    })

    it('test_deselecting_primary_mode_moves_it_to_a_selected_mode', async () => {
        const { state, root } = mountPicker(['aasm', 'aasm_relaxed', 'resmed'], 'aasm')

        await clickMode(root, 'Modes', 'AASM')

        expect(state.modes).toEqual(['aasm_relaxed', 'resmed'])
        expect(state.primaryMode).toBe('aasm_relaxed')
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
