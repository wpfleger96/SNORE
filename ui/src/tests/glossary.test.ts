import { describe, expect, it } from 'vitest'
import { GLOSSARY } from '@/utils/glossary'
import { FIELD_PROVENANCE } from '@/types/provenance.generated'

describe('GLOSSARY provenance', () => {
    it('test_glossary_tier_never_contradicts_generated_field_tier', () => {
        // A glossary key that is also a tagged API field takes its tier from the
        // backend tag; a glossary `provenance` there may only repeat it.
        const contradictions = Object.entries(GLOSSARY)
            .filter(([key, entry]) => key in FIELD_PROVENANCE && entry.provenance !== undefined)
            .filter(([key, entry]) => entry.provenance !== FIELD_PROVENANCE[key])
            .map(
                ([key, entry]) =>
                    `${key}: glossary ${entry.provenance}, field ${FIELD_PROVENANCE[key]}`,
            )

        expect(contradictions).toEqual([])
    })

    it('test_rdi_entry_does_not_claim_rdi_always_exceeds_shown_ahi', () => {
        // RDI is built on SNORE's recount, so on device-headline nights it can sit
        // below the AHI the day view shows.
        const text = `${GLOSSARY.rdi.short} ${GLOSSARY.rdi.long ?? ''}`
        expect(text).not.toMatch(/always ≥ AHI/)
        expect(text).toMatch(/recount/)
    })
})
