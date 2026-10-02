import { describe, expect, it, vi } from 'vitest'
import provenance from '@/types/provenance.json'
import {
    PROVENANCE_TIERS,
    glossaryProvenance,
    provenanceFor,
    provenanceLabel,
    provenanceNote,
} from '@/utils/provenance'

describe('provenanceFor', () => {
    it('test_tiers_match_exported_notes', () => {
        expect([...PROVENANCE_TIERS].sort()).toEqual(Object.keys(provenance.notes).sort())
    })

    it('test_key_resolves_its_schema_tier', () => {
        expect(provenanceFor('DayDetail.ahi_computed')).toBe('derived')
        expect(provenanceFor('DayDetail.ahi')).toBe('device')
        expect(provenanceFor('SessionStatistics.ahi')).toBe('derived')
        expect(provenanceFor('ModeResult.ahi')).toBe('experimental')
    })

    it('test_source_sets_tier_for_fields_with_a_source_sibling', () => {
        expect(provenanceFor('DayDetail.ahi', 'derived')).toBe('derived')
        expect(provenanceFor('DayDetail.ahi', 'device')).toBe('device')
        expect(provenanceFor('EventComparisonDetail.duration', 'machine')).toBe('device')
        expect(provenanceFor('EventComparisonDetail.duration', 'programmatic')).toBe('experimental')
    })

    it('test_null_source_falls_back_to_schema_tier', () => {
        expect(provenanceFor('DayDetail.ahi', null)).toBe('device')
    })

    it('test_source_for_field_without_sibling_is_ignored', () => {
        expect(provenanceFor('DayDetail.ahi_computed', 'device')).toBe('derived')
    })

    it('test_unrecognised_source_warns_and_falls_back', () => {
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
        try {
            expect(provenanceFor('DayDetail.ahi', 'oscar')).toBe('device')
            expect(warn).toHaveBeenCalledWith(
                expect.stringMatching(/unrecognised index_source value "oscar"/),
            )
        } finally {
            warn.mockRestore()
        }
    })
})

describe('glossaryProvenance', () => {
    it('test_glossary_tier', () => {
        expect(glossaryProvenance('sensitivity')).toBe('experimental')
    })

    it('test_glossary_key_without_tier_warns', () => {
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
        expect(glossaryProvenance('ahi')).toBe('device')
        expect(warn).toHaveBeenCalledWith(expect.stringContaining('[glossaryProvenance]'))
        warn.mockRestore()
    })
})

describe('provenance labels', () => {
    it('test_labels_and_notes_per_tier', () => {
        expect(provenanceLabel('device')).toBe('Device-reported')
        expect(provenanceLabel('derived')).toBe('Derived')
        expect(provenanceLabel('experimental')).toBe('Experimental')
        expect(provenanceNote('derived')).toBe('Computed by SNORE from device data.')
    })
})
