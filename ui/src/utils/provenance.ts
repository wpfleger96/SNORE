// Provenance tiers for displayed metrics. The backend tags every metric field
// (snore/provenance.py); `just ui-generate-types` turns those tags into
// @/types/provenance.generated. Views resolve a value's tier with
// provenanceFor() and render it with <ProvenanceMark>.
import type { Component } from 'vue'
import { FlaskConical, Sigma } from '@lucide/vue'
import {
    FIELD_PROVENANCE,
    PROVENANCE_NOTES,
    SCHEMA_FIELD_PROVENANCE,
    type Provenance,
} from '@/types/provenance.generated'
import { GLOSSARY } from '@/utils/glossary'

export type { Provenance }

/** Tiers that get a visible mark; device-reported values stay unmarked. */
export type MarkedProvenance = Exclude<Provenance, 'device'>

const TIERS: readonly string[] = ['device', 'derived', 'experimental'] satisfies Provenance[]

// Per-value source contents that are not tier names: event `source` fields
// say who scored the event.
const SOURCE_VALUE_TIERS: Record<string, Provenance> = {
    machine: 'device',
    programmatic: 'experimental',
}

// Field names whose tier differs by schema (they need `opts.schema`).
const SCHEMA_DEPENDENT_FIELDS = new Set(
    Object.keys(SCHEMA_FIELD_PROVENANCE).map((key) => key.slice(key.indexOf('.') + 1)),
)

export interface ProvenanceOptions {
    /**
     * Content of the field's per-value source sibling (FIELD_PROVENANCE_SOURCE),
     * e.g. `day.index_source` for a day's `ahi`, or an event's `source`.
     */
    source?: string | null
    /**
     * Response schema the value came from (e.g. 'DayDetail'). Required for the
     * few field names whose meaning differs by schema (ahi, oai, cai, hi,
     * duration, duration_hours); ignored otherwise.
     */
    schema?: string
}

/**
 * Tier of the value shown for `field` (an API field name or a GLOSSARY key).
 *
 * Precedence: per-value `opts.source` > generated map (per-schema entry, then
 * bare field name) > GLOSSARY[field].provenance > 'device'.
 */
export function provenanceFor(field: string, opts: ProvenanceOptions = {}): Provenance {
    const { source, schema } = opts
    if (source) {
        if (TIERS.includes(source)) return source as Provenance
        const mapped = SOURCE_VALUE_TIERS[source]
        if (mapped) return mapped
    }
    const perSchema = schema ? SCHEMA_FIELD_PROVENANCE[`${schema}.${field}`] : undefined
    if (perSchema) return perSchema
    if (import.meta.env.DEV && SCHEMA_DEPENDENT_FIELDS.has(field)) {
        console.warn(
            `[provenanceFor] "${field}" has a schema-dependent tier; pass opts.schema` +
                (schema ? ` (no entry for "${schema}.${field}")` : ''),
        )
    }
    return FIELD_PROVENANCE[field] ?? GLOSSARY[field]?.provenance ?? 'device'
}

const LABELS: Record<Provenance, string> = {
    device: 'Device-reported',
    derived: 'Derived',
    experimental: 'Experimental',
}

/** Short tier name: "Device-reported" / "Derived" / "Experimental". */
export function provenanceLabel(provenance: Provenance): string {
    return LABELS[provenance]
}

/** Canonical one-line definition of the tier (from snore.provenance.PROVENANCE_NOTES). */
export function provenanceNote(provenance: Provenance): string {
    return PROVENANCE_NOTES[provenance]
}

/** Icon and theme-aware classes shared by every visible provenance mark. */
export const PROVENANCE_MARK_STYLES: Record<
    MarkedProvenance,
    { icon: Component; iconClass: string; badgeClass: string }
> = {
    derived: {
        icon: Sigma,
        iconClass: 'text-muted-foreground',
        badgeClass: 'border-border bg-muted text-muted-foreground',
    },
    experimental: {
        icon: FlaskConical,
        iconClass: 'text-amber-600 dark:text-amber-400',
        badgeClass:
            'border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-200',
    },
}
