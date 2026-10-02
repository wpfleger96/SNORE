<template>
    <Popover v-if="style" v-model:open="open">
        <PopoverTrigger as-child>
            <button
                type="button"
                :class="[
                    'provenance-mark relative inline-flex shrink-0 items-center justify-center align-middle rounded-sm transition-opacity hover:opacity-80 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring',
                    variant === 'badge'
                        ? `gap-1 rounded-full border px-1.5 py-px text-[10px] font-medium leading-tight ${style.badgeClass}`
                        : `h-4 w-4 ${style.iconClass}`,
                ]"
                :aria-label="`${label}: ${note}`"
                @pointerenter="onTriggerPointerEnter"
                @pointerleave="onPointerLeave"
            >
                <component
                    :is="style.icon"
                    :class="variant === 'badge' ? 'h-3 w-3' : 'h-3.5 w-3.5'"
                    aria-hidden="true"
                />
                <span v-if="variant === 'badge'">{{ label }}</span>
            </button>
        </PopoverTrigger>
        <PopoverContent
            class="w-72"
            @pointerenter="cancelClose"
            @pointerleave="onPointerLeave"
            @open-auto-focus="onOpenAutoFocus"
        >
            <PopoverHeader>
                <PopoverTitle class="flex items-center gap-1.5">
                    <component :is="style.icon" :class="['h-4 w-4', style.iconClass]" />
                    {{ label }}
                </PopoverTitle>
            </PopoverHeader>
            <PopoverDescription>{{ note }}</PopoverDescription>
        </PopoverContent>
    </Popover>
</template>

<script setup lang="ts">
// Inline provenance marker for a metric value or label: renders nothing for
// device-reported values, a muted Sigma for derived, an amber flask for
// experimental. Resolve the tier with provenanceFor() from @/utils/provenance.
import { computed } from 'vue'
import {
    Popover,
    PopoverContent,
    PopoverDescription,
    PopoverHeader,
    PopoverTitle,
    PopoverTrigger,
} from '@/components/ui/popover'
import { useHoverPopover } from '@/composables/useHoverPopover'
import {
    PROVENANCE_MARK_STYLES,
    provenanceLabel,
    provenanceNote,
    type Provenance,
} from '@/utils/provenance'

const props = withDefaults(
    defineProps<{
        provenance: Provenance
        // 'icon' sits next to a label; 'badge' is a small labelled pill.
        variant?: 'icon' | 'badge'
    }>(),
    { variant: 'icon' },
)

const style = computed(() =>
    props.provenance === 'device' ? null : PROVENANCE_MARK_STYLES[props.provenance],
)
const label = computed(() => provenanceLabel(props.provenance))
const note = computed(() => provenanceNote(props.provenance))

const { open, onTriggerPointerEnter, onPointerLeave, cancelClose, onOpenAutoFocus } =
    useHoverPopover()
</script>

<style scoped>
/* Enlarge the touch hit area to the shared tap target without changing the
   inline layout footprint of the mark. */
@media (max-width: 767.98px) {
    .provenance-mark::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: max(100%, var(--tap-target));
        height: max(100%, var(--tap-target));
        transform: translate(-50%, -50%);
    }
}
</style>
