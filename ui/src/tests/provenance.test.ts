import { describe, expect, it, vi } from 'vitest'
import { FIELD_PROVENANCE, SCHEMA_FIELD_PROVENANCE } from '@/types/provenance.generated'
import {
    glossaryProvenance,
    provenanceFor,
    provenanceLabel,
    provenanceNote,
} from '@/utils/provenance'

function expectWarning(run: () => void, message: string | RegExp) {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
    try {
        run()
        expect(warn).toHaveBeenCalledWith(expect.stringMatching(message))
    } finally {
        warn.mockRestore()
    }
}

describe('provenanceFor', () => {
    it('test_generated_field_map_resolves_tagged_fields', () => {
        expect(FIELD_PROVENANCE.ahi_computed).toBe('derived')
        expect(provenanceFor('ahi_computed')).toBe('derived')
    })

    it('test_source_sets_tier_for_fields_with_a_source_sibling', () => {
        expect(provenanceFor('ahi', { schema: 'DayDetail', source: 'derived' })).toBe('derived')
        expect(provenanceFor('ahi', { schema: 'DayDetail', source: 'device' })).toBe('device')
    })

    it('test_event_source_values_map_to_tiers', () => {
        const opts = { schema: 'EventComparisonDetail' } as const
        expect(provenanceFor('duration', { ...opts, source: 'machine' })).toBe('device')
        expect(provenanceFor('duration', { ...opts, source: 'programmatic' })).toBe('experimental')
    })

    it('test_null_source_falls_back_to_schema_tier', () => {
        expect(provenanceFor('ahi', { schema: 'DayDetail', source: null })).toBe('device')
    })

    it('test_source_for_field_without_sibling_is_ignored_with_warning', () => {
        expectWarning(() => {
            expect(provenanceFor('ahi_computed', { source: 'device' })).toBe('derived')
        }, /no per-value source sibling/)
    })

    it('test_unrecognised_source_warns_and_falls_back', () => {
        expectWarning(() => {
            expect(provenanceFor('ahi', { schema: 'DayDetail', source: 'oscar' })).toBe('device')
        }, /unrecognised index_source value "oscar"/)
    })

    it('test_schema_selects_tier_for_schema_dependent_fields', () => {
        expect(SCHEMA_FIELD_PROVENANCE['DayDetail.ahi']).toBe('device')
        expect(provenanceFor('ahi', { schema: 'DayDetail' })).toBe('device')
        expect(provenanceFor('ahi', { schema: 'SessionStatistics' })).toBe('derived')
        expect(provenanceFor('ahi', { schema: 'ModeResult' })).toBe('experimental')
    })

    it('test_schema_dependent_field_without_schema_warns_and_uses_weakest_tier', () => {
        expectWarning(() => {
            expect(provenanceFor('ahi')).toBe('experimental')
            expect(provenanceFor('oai')).toBe('derived')
        }, /pass opts.schema/)
    })

    it('test_schema_without_entry_for_ambiguous_field_uses_weakest_tier', () => {
        expectWarning(() => {
            expect(provenanceFor('hi', { schema: 'ModeResult' })).toBe('derived')
        }, /no entry for "ModeResult.hi"/)
    })

    it('test_untagged_field_warns_and_defaults_to_device', () => {
        expectWarning(() => {
            expect(provenanceFor('not_a_field_xyz')).toBe('device')
        }, /not a tagged API field/)
    })

    it('test_allowlisted_untagged_field_is_device_without_warning', () => {
        expect(provenanceFor('total_sessions')).toBe('device')
    })

    it('test_glossary_keys_are_not_field_names', () => {
        // provenanceFor reads only the generated maps; glossary tiers go through
        // glossaryProvenance.
        expectWarning(() => {
            expect(provenanceFor('sensitivity')).toBe('device')
        }, /not a tagged API field/)
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
