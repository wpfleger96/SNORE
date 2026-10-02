// @vitest-environment node
import { describe, expect, it } from 'vitest'
import { ahiColorClass, ahiTier } from '@/utils/ahiScale'
import { ahiClass } from '@/utils/formatting'

const BOUNDARIES: [number, string][] = [
    [0, 'good'],
    [4.9, 'good'],
    [5, 'mild'],
    [9.9, 'mild'],
    [10, 'moderate'],
    [12, 'moderate'],
    [14.9, 'moderate'],
    [15, 'severe'],
    [42, 'severe'],
]

describe('ahiTier', () => {
    it.each(BOUNDARIES)('test_ahi_%s_maps_to_%s', (ahi, tier) => {
        expect(ahiTier(ahi)).toBe(tier)
    })

    it('test_nullish_has_no_tier', () => {
        expect(ahiTier(null)).toBeNull()
        expect(ahiTier(undefined)).toBeNull()
    })
})

describe('ahiColorClass', () => {
    it.each(BOUNDARIES)('test_ahi_%s_maps_to_cell_%s', (ahi, tier) => {
        expect(ahiColorClass(ahi)).toBe(`cell--${tier}`)
    })

    it('test_null_maps_to_empty_cell', () => {
        expect(ahiColorClass(null)).toBe('cell--empty')
    })
})

describe('ahiClass', () => {
    it.each(BOUNDARIES)('test_ahi_%s_maps_to_text_class_%s', (ahi, tier) => {
        expect(ahiClass(ahi)).toBe(`ahi-${tier}`)
    })

    it('test_nullish_maps_to_no_class', () => {
        expect(ahiClass(null)).toBe('')
        expect(ahiClass(undefined)).toBe('')
    })
})
