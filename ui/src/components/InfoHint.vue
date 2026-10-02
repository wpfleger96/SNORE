<template>
    <Popover v-model:open="open">
        <PopoverTrigger as-child>
            <button
                type="button"
                class="inline-flex items-center justify-center align-middle text-muted-foreground hover:text-foreground transition-colors rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring h-4 w-4 shrink-0"
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
            <p v-if="tier" class="info-hint-tier flex items-start gap-1.5 text-xs">
                <component
                    :is="tier.icon"
                    :class="['mt-px h-3.5 w-3.5 shrink-0', tier.iconClass]"
                    aria-hidden="true"
                />
                <span>
                    <span class="font-medium">{{ tier.label }}</span>
                    <span class="text-muted-foreground"> — {{ tier.note }}</span>
                </span>
            </p>
        </PopoverContent>
    </Popover>
</template>

<script setup lang="ts">
import { computed, watchEffect } from 'vue'
import { Info } from '@lucide/vue'
import { GLOSSARY } from '@/utils/glossary'
import {
    PROVENANCE_MARK_STYLES,
    provenanceLabel,
    provenanceNote,
    type Provenance,
} from '@/utils/provenance'
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
    // Tier of the value this hint explains (resolve with provenanceFor());
    // derived/experimental add a tier line to the popover.
    provenance?: Provenance
}>()

const entry = computed(() => (props.glossaryKey ? (GLOSSARY[props.glossaryKey] ?? null) : null))

const resolvedLabel = computed(() => props.label ?? entry.value?.label ?? '')
const resolvedShort = computed(() => props.short ?? entry.value?.short ?? '')
const resolvedLong = computed(() => props.long ?? entry.value?.long ?? '')

const tier = computed(() => {
    const p = props.provenance
    if (!p || p === 'device') return null
    return {
        ...PROVENANCE_MARK_STYLES[p],
        label: provenanceLabel(p),
        note: provenanceNote(p),
    }
})

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
