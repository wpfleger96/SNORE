<template>
    <Popover v-model:open="open">
        <PopoverTrigger as-child>
            <button
                type="button"
                class="info-hint relative inline-flex items-center justify-center align-middle text-muted-foreground hover:text-foreground transition-colors rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring h-4 w-4 shrink-0"
                :aria-label="`More information about ${resolvedLabel}`"
                @pointerenter="onTriggerPointerEnter"
                @pointerleave="onPointerLeave"
            >
                <Info class="h-3.5 w-3.5" />
            </button>
        </PopoverTrigger>
        <PopoverContent
            class="w-72"
            @pointerenter="cancelClose"
            @pointerleave="onPointerLeave"
            @open-auto-focus="onOpenAutoFocus"
        >
            <PopoverHeader>
                <PopoverTitle>{{ resolvedLabel }}</PopoverTitle>
            </PopoverHeader>
            <slot v-if="$slots.default" />
            <template v-else>
                <PopoverDescription v-if="resolvedShort">{{ resolvedShort }}</PopoverDescription>
                <p v-if="resolvedLong" class="text-xs text-muted-foreground">{{ resolvedLong }}</p>
            </template>
        </PopoverContent>
    </Popover>
</template>

<script setup lang="ts">
import { computed, watchEffect } from 'vue'
import { Info } from '@lucide/vue'
import { GLOSSARY } from '@/utils/glossary'
import { useHoverPopover } from '@/composables/useHoverPopover'
import {
    Popover,
    PopoverContent,
    PopoverDescription,
    PopoverHeader,
    PopoverTitle,
    PopoverTrigger,
} from '@/components/ui/popover'

const props = defineProps<{
    glossaryKey?: string
    label?: string
    short?: string
    long?: string
}>()

const entry = computed(() => (props.glossaryKey ? (GLOSSARY[props.glossaryKey] ?? null) : null))

const resolvedLabel = computed(() => props.label ?? entry.value?.label ?? '')
const resolvedShort = computed(() => props.short ?? entry.value?.short ?? '')
const resolvedLong = computed(() => props.long ?? entry.value?.long ?? '')

// Click/tap toggles; mouse hover opens and closes after a short grace period.
const { open, onTriggerPointerEnter, onPointerLeave, cancelClose, onOpenAutoFocus } =
    useHoverPopover()

if (import.meta.env.DEV) {
    watchEffect(() => {
        if (props.glossaryKey && !entry.value && !props.label) {
            console.warn(`[InfoHint] No glossary entry found for key: "${props.glossaryKey}"`)
        }
    })
}
</script>

<style scoped>
/* Mobile tap target (see the matching rule in ProvenanceMark.vue). The hit area
   extends rightward from just left of the icon, and a mark's extends leftward,
   so a <ProvenanceMark> placed before an InfoHint never steals its taps. */
@media (max-width: 767.98px) {
    .info-hint::before {
        content: '';
        position: absolute;
        top: 50%;
        left: -0.125rem;
        width: max(100%, var(--tap-target));
        height: max(100%, var(--tap-target));
        transform: translateY(-50%);
    }
}
</style>
