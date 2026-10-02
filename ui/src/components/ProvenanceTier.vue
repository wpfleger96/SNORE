<template>
    <div class="flex items-start gap-1.5 text-xs">
        <component
            :is="style.icon"
            :class="['mt-px h-3.5 w-3.5 shrink-0', style.iconClass]"
            aria-hidden="true"
        />
        <span>
            <span class="font-medium text-foreground">{{ provenanceLabel(tier) }}</span>
            <span class="text-muted-foreground"> — {{ provenanceNote(tier) }}</span>
        </span>
    </div>
</template>

<script setup lang="ts">
// One tier's icon, name, and definition: the single rendering shared by the
// <ProvenanceMark> popover and <ProvenanceLegend>.
import { computed } from 'vue'
import {
    PROVENANCE_MARK_STYLES,
    provenanceLabel,
    provenanceNote,
    type MarkedProvenance,
} from '@/utils/provenance'

const props = defineProps<{ tier: MarkedProvenance }>()

const style = computed(() => PROVENANCE_MARK_STYLES[props.tier])
</script>
