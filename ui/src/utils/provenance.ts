// Provenance tiers for displayed metrics. The backend tags every metric field
// (snore/provenance.py); `just ui-generate-types` turns those tags into
// @/types/provenance.generated. Views resolve an API field's tier with
// provenanceFor(), a non-API metric's tier with glossaryProvenance(), and
// render it with <ProvenanceMark>.
import type { Component } from 'vue'
import { FlaskConical, Sigma } from '@lucide/vue'
import type { components } from '@/types/generated'
import {
    FIELD_PROVENANCE,
    PROVENANCE_NOTES,
    PROVENANCE_SOURCE_FIELDS,
    PROVENANCE_TIERS,
    SCHEMA_FIELD_PROVENANCE,
    type Provenance,
} from '@/types/provenance.generated'
import { GLOSSARY } from '@/utils/glossary'

export { PROVENANCE_TIERS, type Provenance }

/** Tiers that get a visible mark; device-reported values stay unmarked. */
export type MarkedProvenance = Exclude<Provenance, 'device'>

/** Name of an API response schema (a key of `components['schemas']`). */
export type SchemaName = keyof components['schemas']

/**
 * Displayed fields that are deliberately untagged, so provenanceFor() returns
 * 'device' (unmarked) for them without warning. Add a name only with a reason.
 */
export const UNTAGGED_DISPLAY_FIELDS: ReadonlySet<string> = new Set([
    // Validation-run bookkeeping counts shown in the run comparison
    // (utils/validationMetrics); the backend leaves them untagged as non-metrics
    // (_NON_METRIC_MODEL_FIELDS in tests/unit/test_provenance_tags.py).
    'n_with_apple_bd',
    'sessions_compared',
    'total_nights',
    'total_sessions',
])

// Content of a per-value source sibling -> tier. `index_source` holds tier
// names; an event's `source` says who scored it.
const SOURCE_TIERS: Record<string, Provenance> = {
    ...(Object.fromEntries(PROVENANCE_TIERS.map((t) => [t, t])) as Record<Provenance, Provenance>),
    machine: 'device',
    programmatic: 'experimental',
}

const weaker = (a: Provenance, b: Provenance): Provenance =>
    PROVENANCE_TIERS.indexOf(a) >= PROVENANCE_TIERS.indexOf(b) ? a : b

// Field names whose tier differs by schema -> the weakest tier any schema
// gives them; the fallback when the caller's schema has no entry, so a missing
// schema over-marks rather than hiding a mark.
const SCHEMA_DEPENDENT_WEAKEST: Record<string, Provenance> = {}
for (const [key, tier] of Object.entries(SCHEMA_FIELD_PROVENANCE)) {
    const field = key.slice(key.indexOf('.') + 1)
    const prev = SCHEMA_DEPENDENT_WEAKEST[field]
    SCHEMA_DEPENDENT_WEAKEST[field] = prev ? weaker(prev, tier) : tier
}

function warn(message: string): void {
    if (import.meta.env.DEV) console.warn(`[provenanceFor] ${message}`)
}

export interface ProvenanceOptions {
    /**
     * Response schema the value came from (e.g. 'DayDetail'). Required for the
     * few field names whose meaning differs by schema (ahi, oai, cai, hi,
     * duration, duration_hours); harmless otherwise.
     */
    schema?: SchemaName
    /**
     * Content of the field's per-value source sibling (PROVENANCE_SOURCE_FIELDS),
     * e.g. `day.index_source` for DayDetail.ahi, or an event's `source` for
     * EventComparisonDetail.duration. Ignored (with a warning) for fields that
     * have no source sibling.
     */
    source?: string | null
}

/**
 * Tier of the value an API response field carries.
 *
 * Precedence: per-value `opts.source` (only for fields with a source sibling)
 * > per-schema entry > bare field name > 'device'. An ambiguous field without
 * a matching schema entry resolves to its weakest tier.
 */
export function provenanceFor(field: string, opts: ProvenanceOptions = {}): Provenance {
    const { schema, source } = opts
    const qualified = schema ? `${schema}.${field}` : undefined
    const sourceField =
        (qualified && PROVENANCE_SOURCE_FIELDS[qualified]) ?? PROVENANCE_SOURCE_FIELDS[field]
    if (source != null) {
        const tier = SOURCE_TIERS[source]
        if (!sourceField) warn(`"${field}" has no per-value source sibling; source ignored`)
        else if (!tier) warn(`unrecognised ${sourceField} value "${source}" for "${field}"`)
        else return tier
    }
    const perSchema = qualified ? SCHEMA_FIELD_PROVENANCE[qualified] : undefined
    if (perSchema) return perSchema
    const weakest = SCHEMA_DEPENDENT_WEAKEST[field]
    if (weakest) {
        warn(
            `"${field}" has a schema-dependent tier; pass opts.schema` +
                (schema ? ` (no entry for "${qualified}")` : ''),
        )
        return weakest
    }
    const tier = FIELD_PROVENANCE[field]
    if (tier) return tier
    if (!UNTAGGED_DISPLAY_FIELDS.has(field)) {
        warn(`"${field}" is not a tagged API field (use glossaryProvenance for non-API metrics)`)
    }
    return 'device'
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
export function provenanceLabel(provenance: Provenance): string {
    return LABELS[provenance]
}

/** Canonical one-line definition of the tier (from snore.provenance.PROVENANCE_NOTES). */
export function provenanceNote(provenance: Provenance): string {
    return PROVENANCE_NOTES[provenance]
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
