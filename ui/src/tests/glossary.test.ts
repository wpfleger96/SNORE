import { describe, expect, it } from 'vitest'
import { GLOSSARY } from '@/utils/glossary'

// App source (not tests), to check which glossary tiers a call site reads.
const SOURCES = Object.entries(
    import.meta.glob<string>('/src/**/*.{ts,vue}', {
        query: '?raw',
        import: 'default',
        eager: true,
    }),
)
    .filter(([path]) => !path.startsWith('/src/tests/') && !path.includes('generated'))
    .map(([, source]) => source)
    .join('\n')

describe('GLOSSARY provenance', () => {
    it('test_every_glossary_tier_is_read_by_a_call_site', () => {
        // A `provenance` nothing reads via glossaryProvenance() is dead and drifts.
        const unread = Object.entries(GLOSSARY)
            .filter(([, entry]) => entry.provenance !== undefined)
            .map(([key]) => key)
            .filter((key) => !SOURCES.includes(`glossaryProvenance('${key}')`))

        expect(unread).toEqual([])
    })

    it('test_every_glossary_provenance_call_has_a_tier', () => {
        const called = [...SOURCES.matchAll(/glossaryProvenance\('(\w+)'\)/g)].map((m) => m[1])
        const missing = called.filter((key) => GLOSSARY[key]?.provenance === undefined)

        expect(called.length).toBeGreaterThan(0)
        expect(missing).toEqual([])
    })
})

describe('GLOSSARY text', () => {
    it('test_rdi_entry_says_rdi_never_below_mode_ahi', () => {
        // The UI shows ModeResult.rdi: the mode's AHI plus its RERAs per hour.
        const text = `${GLOSSARY.rdi.short} ${GLOSSARY.rdi.long ?? ''}`
        expect(text).toMatch(/never below that mode's AHI/)
        expect(text).not.toMatch(/below the AHI shown/)
    })

    it('test_ahi_entry_describes_period_values_as_weighted_averages', () => {
        const text = GLOSSARY.ahi.long ?? ''
        expect(text).toMatch(/usage-hours-weighted averages/)
        expect(text).toMatch(/no disabled sessions/)
        expect(text).not.toMatch(/period, and trend AHIs are SNORE recounts/)
    })

    it('test_labels_are_unique_outside_device_settings', () => {
        // Device-setting entries (setting_*) reuse metric names by design; the two
        // I:E Ratio entries are one quantity from two sources (waveform, STR).
        const seen = new Map<string, string>()
        const duplicates: string[] = []
        for (const [key, entry] of Object.entries(GLOSSARY)) {
            if (key.startsWith('setting_') || entry.label === 'I:E Ratio') continue
            const prev = seen.get(entry.label)
            if (prev) duplicates.push(`${entry.label}: ${prev}, ${key}`)
            seen.set(entry.label, key)
        }

        expect(duplicates).toEqual([])
    })

    it('test_user_facing_text_uses_device_and_snore_terms', () => {
        const text = Object.values(GLOSSARY)
            .map((e) => `${e.label} ${e.short} ${e.long ?? ''}`)
            .join('\n')

        expect(text).not.toMatch(/\bmachine\b|machine-|programmatic/i)
    })
})
