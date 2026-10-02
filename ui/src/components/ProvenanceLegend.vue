<template>
    <div class="space-y-1.5 text-xs text-muted-foreground">
        <dl class="space-y-1.5" aria-label="Data provenance legend">
            <div v-for="tier in TIERS" :key="tier" class="flex items-start gap-1.5">
                <dt class="flex shrink-0 items-center gap-1 font-medium text-foreground">
                    <component
                        :is="PROVENANCE_MARK_STYLES[tier].icon"
                        :class="['h-3.5 w-3.5', PROVENANCE_MARK_STYLES[tier].iconClass]"
                        aria-hidden="true"
                    />
                    {{ provenanceLabel(tier) }}
                </dt>
                <dd>{{ provenanceNote(tier) }}</dd>
            </div>
        </dl>
        <p>Unmarked values are device-reported.</p>
    </div>
</template>

<script setup lang="ts">
// Key for the marks <ProvenanceMark> renders next to metric values.
import {
    PROVENANCE_MARK_STYLES,
    provenanceLabel,
    provenanceNote,
    type MarkedProvenance,
} from '@/utils/provenance'

const TIERS: MarkedProvenance[] = ['derived', 'experimental']
</script>
