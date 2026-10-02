// Provenance tiers for displayed metrics. The backend tags every metric field
// (snore/provenance.py); `just ui-generate-types` exports those tags to
// @/types/provenance.json, keyed `Schema.field`. Views resolve an API field's
// tier with provenanceFor(), a non-API metric's tier with glossaryProvenance(),
// and render it with <ProvenanceMark>.
import type { Component } from 'vue'
import { FlaskConical, Sigma } from '@lucide/vue'
import provenance from '@/types/provenance.json'
import { GLOSSARY } from '@/utils/glossary'

/** Every tier, strongest (device) to weakest (experimental). */
export const PROVENANCE_TIERS = ['device', 'derived', 'experimental'] as const

export type Provenance = (typeof PROVENANCE_TIERS)[number]

/** Tiers that get a visible mark; device-reported values stay unmarked. */
export type MarkedProvenance = Exclude<Provenance, 'device'>

/** A tagged API field, `Schema.field` (e.g. 'DayDetail.ahi'). */
export type ProvenanceKey = keyof typeof provenance.fields

const FIELDS = provenance.fields as Record<ProvenanceKey, Provenance>
const SOURCES: Partial<Record<ProvenanceKey, string>> = provenance.sources

// Content of a per-value source sibling -> tier. `index_source` holds tier
// names; an event's `source` says who scored it.
const SOURCE_TIERS: Record<string, Provenance> = {
    device: 'device',
    derived: 'derived',
    experimental: 'experimental',
    machine: 'device',
    programmatic: 'experimental',
}

/**
 * Tier of the value an API field carries. `source` is the content of the
 * field's per-value source sibling (`provenance.json` `sources`, e.g.
 * `day.index_source` for 'DayDetail.ahi'); it is ignored for other fields.
 */
export function provenanceFor(key: ProvenanceKey, source?: string | null): Provenance {
    const sourceField = SOURCES[key]
    if (sourceField && source != null) {
        const tier = SOURCE_TIERS[source]
        if (tier) return tier
        if (import.meta.env.DEV) {
            console.warn(`[provenanceFor] unrecognised ${sourceField} value "${source}" for ${key}`)
        }
    }
    return FIELDS[key]
}

/**
 * Tier of a displayed metric with no backing API field, from its GLOSSARY
 * entry's `provenance`.
 */
export function glossaryProvenance(key: string): Provenance {
    const tier = GLOSSARY[key]?.provenance
    if (!tier && import.meta.env.DEV) {
        console.warn(`[glossaryProvenance] glossary entry "${key}" has no provenance`)
    }
    return tier ?? 'device'
}

const LABELS: Record<Provenance, string> = {
    device: 'Device-reported',
    derived: 'Derived',
    experimental: 'Experimental',
}

/** Short tier name: "Device-reported" / "Derived" / "Experimental". */
export function provenanceLabel(tier: Provenance): string {
    return LABELS[tier]
}

/** Canonical one-line definition of the tier (from snore.provenance.PROVENANCE_NOTES). */
export function provenanceNote(tier: Provenance): string {
    return provenance.notes[tier]
}

/**
 * Icon and theme-aware classes shared by every visible provenance mark.
 * `toneClass` is the full-surface treatment (border, background, text) used by
 * ExperimentalBanner.
 */
export const PROVENANCE_MARK_STYLES: Record<
    MarkedProvenance,
    { icon: Component; iconClass: string; toneClass: string }
> = {
    derived: {
        icon: Sigma,
        iconClass: 'text-muted-foreground',
        toneClass: 'border-border bg-muted text-muted-foreground',
    },
    experimental: {
        icon: FlaskConical,
        iconClass: 'text-amber-600 dark:text-amber-400',
        toneClass:
            'border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-200',
    },
}
