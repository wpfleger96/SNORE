<template>
    <div class="space-y-1.5 text-xs text-muted-foreground">
        <ul class="space-y-1.5" aria-label="Data provenance legend">
            <li v-for="tier in MARKED_TIERS" :key="tier"><ProvenanceTier :tier="tier" /></li>
        </ul>
        <p>Unmarked values are device-reported.</p>
    </div>
</template>

<script setup lang="ts">
// Key for the marks <ProvenanceMark> renders next to metric values.
import ProvenanceTier from '@/components/ProvenanceTier.vue'
import { PROVENANCE_TIERS, type MarkedProvenance } from '@/utils/provenance'

const MARKED_TIERS = PROVENANCE_TIERS.filter((t): t is MarkedProvenance => t !== 'device')
</script>
