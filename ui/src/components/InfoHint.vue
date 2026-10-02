<template>
    <Popover v-model:open="open">
        <PopoverTrigger as-child>
            <button
                type="button"
                class="inline-flex items-center justify-center align-middle text-muted-foreground hover:text-foreground transition-colors rounded-sm focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring h-4 w-4 shrink-0"
                :aria-label="`More information about ${resolvedLabel}`"
                @pointerenter="onTriggerPointerEnter"
                @pointerleave="onTriggerPointerLeave"
            >
                <Info class="h-3.5 w-3.5" />
            </button>
        </PopoverTrigger>
        <PopoverContent
            class="w-72"
            @pointerenter="cancelClose"
            @pointerleave="onContentPointerLeave"
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
import { computed, onBeforeUnmount, ref, watchEffect } from 'vue'
import { Info } from '@lucide/vue'
import { GLOSSARY } from '@/utils/glossary'
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

// Controlled open state lets native click-toggling and mouse hover cooperate.
const open = ref(false)

// Grace period so moving the pointer from the trigger into the portal-rendered
// content (across the gap between them) does not close the popover.
const CLOSE_DELAY_MS = 150
let closeTimer: ReturnType<typeof setTimeout> | null = null

function cancelClose() {
    if (closeTimer !== null) {
        clearTimeout(closeTimer)
        closeTimer = null
    }
}

function scheduleClose() {
    cancelClose()
    closeTimer = setTimeout(() => {
        open.value = false
        closeTimer = null
    }, CLOSE_DELAY_MS)
}

// Hover behavior is mouse-only: touch taps fire pointerenter before click, so
// guarding on pointerType keeps tap-to-toggle intact for touch users.
function onTriggerPointerEnter(event: PointerEvent) {
    if (event.pointerType !== 'mouse') return
    cancelClose()
    open.value = true
}

function onTriggerPointerLeave(event: PointerEvent) {
    if (event.pointerType !== 'mouse') return
    scheduleClose()
}

function onContentPointerLeave(event: PointerEvent) {
    if (event.pointerType !== 'mouse') return
    scheduleClose()
}

// Never steal keyboard focus on open; focus stays on the trigger. Click and
// keyboard users still get Escape-to-close and click-outside dismissal.
function onOpenAutoFocus(event: Event) {
    event.preventDefault()
}

onBeforeUnmount(cancelClose)

if (import.meta.env.DEV) {
    watchEffect(() => {
        if (props.glossaryKey && !entry.value && !props.label) {
            console.warn(`[InfoHint] No glossary entry found for key: "${props.glossaryKey}"`)
        }
    })
}
</script>
