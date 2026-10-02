<template>
    <Popover v-if="style && tier" v-model:open="open">
        <PopoverTrigger as-child>
            <button
                type="button"
                :class="[
                    'provenance-mark relative inline-flex h-4 w-4 shrink-0 items-center justify-center align-middle rounded-sm transition-opacity hover:opacity-80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                    style.iconClass,
                ]"
                :aria-label="`${provenanceLabel(tier)}: ${provenanceNote(tier)}`"
                @pointerenter="onTriggerPointerEnter"
                @pointerleave="onPointerLeave"
            >
                <component :is="style.icon" class="h-3.5 w-3.5" aria-hidden="true" />
            </button>
        </PopoverTrigger>
        <PopoverContent
            class="w-72"
            @pointerenter="cancelClose"
            @pointerleave="onPointerLeave"
            @open-auto-focus="onOpenAutoFocus"
        >
            <ProvenanceTier :tier="tier" />
        </PopoverContent>
    </Popover>
</template>

<script setup lang="ts">
// Inline provenance marker for a metric value or label: renders nothing for
// device-reported values, a muted Sigma for derived, an amber flask for
// experimental. Resolve the tier with provenanceFor() / glossaryProvenance()
// from @/utils/provenance.
import { computed } from 'vue'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import ProvenanceTier from '@/components/ProvenanceTier.vue'
import { useHoverPopover } from '@/composables/useHoverPopover'
import {
    PROVENANCE_MARK_STYLES,
    provenanceLabel,
    provenanceNote,
    type MarkedProvenance,
    type Provenance,
} from '@/utils/provenance'

const props = defineProps<{ provenance: Provenance }>()

const tier = computed<MarkedProvenance | null>(() =>
    props.provenance === 'device' ? null : props.provenance,
)
const style = computed(() => (tier.value ? PROVENANCE_MARK_STYLES[tier.value] : null))

const { open, onTriggerPointerEnter, onPointerLeave, cancelClose, onOpenAutoFocus } =
    useHoverPopover()
</script>

<style scoped>
/* Mobile tap target without changing the mark's inline footprint. The hit area
   extends leftward from just right of the icon, and an InfoHint's extends
   rightward, so a mark placed before an InfoHint never steals its taps. */
@media (max-width: 767.98px) {
    .provenance-mark::before {
        content: '';
        position: absolute;
        top: 50%;
        right: -0.125rem;
        width: max(100%, var(--tap-target));
        height: max(100%, var(--tap-target));
        transform: translateY(-50%);
    }
}
</style>
