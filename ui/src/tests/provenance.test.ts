import { describe, expect, it, vi } from 'vitest'
import { FIELD_PROVENANCE, SCHEMA_FIELD_PROVENANCE } from '@/types/provenance.generated'
import { provenanceFor, provenanceLabel, provenanceNote } from '@/utils/provenance'

describe('provenanceFor', () => {
    it('test_generated_field_map_resolves_tagged_fields', () => {
        expect(FIELD_PROVENANCE.ahi_computed).toBe('derived')
        expect(provenanceFor('ahi_computed')).toBe('derived')
    })

    it('test_explicit_tier_source_overrides_generated_map', () => {
        expect(provenanceFor('ahi_computed', { source: 'device' })).toBe('device')
        expect(provenanceFor('ahi', { schema: 'DayDetail', source: 'derived' })).toBe('derived')
    })

    it('test_event_source_values_map_to_tiers', () => {
        expect(provenanceFor('duration', { schema: 'AnalysisEvent', source: 'machine' })).toBe(
            'device',
        )
        expect(provenanceFor('duration', { schema: 'AnalysisEvent', source: 'programmatic' })).toBe(
            'experimental',
        )
    })

    it('test_null_or_unknown_source_falls_back_to_map', () => {
        expect(provenanceFor('ahi_computed', { source: null })).toBe('derived')
        expect(provenanceFor('ahi_computed', { source: 'oscar' })).toBe('derived')
    })

    it('test_schema_selects_tier_for_schema_dependent_fields', () => {
        expect(SCHEMA_FIELD_PROVENANCE['DayDetail.ahi']).toBe('device')
        expect(provenanceFor('ahi', { schema: 'DayDetail' })).toBe('device')
        expect(provenanceFor('ahi', { schema: 'SessionStatistics' })).toBe('derived')
        expect(provenanceFor('ahi', { schema: 'ModeResult' })).toBe('experimental')
    })

    it('test_schema_dependent_field_without_schema_warns_and_defaults', () => {
        const warn = vi.spyOn(console, 'warn').mockImplementation(() => {})
        expect(provenanceFor('ahi')).toBe('device')
        expect(warn).toHaveBeenCalledWith(expect.stringContaining('pass opts.schema'))
        warn.mockRestore()
    })

    it('test_glossary_entry_is_fallback_for_untagged_keys', () => {
        expect(FIELD_PROVENANCE.rera_proxy).toBeUndefined()
        expect(provenanceFor('rera_proxy')).toBe('experimental')
    })

    it('test_unknown_field_defaults_to_device', () => {
        expect(provenanceFor('not_a_field_xyz')).toBe('device')
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
