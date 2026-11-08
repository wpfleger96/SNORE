import { describe, expect, it } from 'vitest'
import { effectivePrimaryMode, modeLabel } from '@/utils/analysisModes'

describe('effectivePrimaryMode', () => {
    it('test_selected_pick_is_kept', () => {
        expect(effectivePrimaryMode(['aasm', 'resmed'], 'resmed')).toBe('resmed')
    })

    it('test_pick_not_selected_falls_back_to_default', () => {
        expect(effectivePrimaryMode(['aasm', 'aasm_relaxed'], 'resmed')).toBe('aasm')
    })

    it('test_without_pick_aasm_is_the_default_when_selected', () => {
        expect(effectivePrimaryMode(['resmed', 'aasm'], null)).toBe('aasm')
    })

    it('test_without_aasm_the_first_selected_mode_in_canonical_order_is_the_default', () => {
        expect(effectivePrimaryMode(['resmed', 'aasm_relaxed'], null)).toBe('aasm_relaxed')
    })

    it('test_no_selected_modes_has_no_primary', () => {
        expect(effectivePrimaryMode([], 'aasm')).toBeUndefined()
    })
})

describe('modeLabel', () => {
    it('test_known_modes_get_display_labels', () => {
        expect(modeLabel('aasm_relaxed')).toBe('AASM Relaxed')
    })

    it('test_unknown_mode_falls_back_to_its_key', () => {
        expect(modeLabel('experimental_v2')).toBe('experimental_v2')
        expect(modeLabel('toString')).toBe('toString')
    })
})
