<template>
    <div class="mode-picker">
        <div>
            <span :id="modesLabelId" class="text-sm font-medium">Modes</span>
            <ToggleGroup
                type="multiple"
                variant="outline"
                class="picker-toggles mt-1"
                :aria-labelledby="modesLabelId"
                :model-value="modes"
                @update:model-value="setModes"
            >
                <ToggleGroupItem
                    v-for="option in MODE_OPTIONS"
                    :key="option.value"
                    :value="option.value"
                >
                    {{ option.label }}
                </ToggleGroupItem>
            </ToggleGroup>
            <p v-if="!modes.length" class="mt-1 text-xs text-destructive">
                Select at least one mode.
            </p>
            <p class="mt-1 text-xs text-muted-foreground">
                Each selected mode runs separately and appears as its own row in the comparison.
            </p>
        </div>
        <Collapsible v-if="modes.length > 1" v-slot="{ open }">
            <CollapsibleTrigger as-child>
                <button
                    type="button"
                    class="advanced-trigger flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
                >
                    <ChevronRight
                        class="h-4 w-4 transition-transform"
                        :class="{ 'rotate-90': open }"
                    />
                    Advanced: RERA flags from {{ modeLabel(primaryMode) }}
                </button>
            </CollapsibleTrigger>
            <CollapsibleContent>
                <div class="mt-2">
                    <span :id="primaryLabelId" class="text-sm font-medium">
                        Mode for stored RERA flags
                    </span>
                    <ToggleGroup
                        type="single"
                        variant="outline"
                        class="picker-toggles mt-1"
                        :aria-labelledby="primaryLabelId"
                        :model-value="primaryMode"
                        @update:model-value="pickPrimary"
                    >
                        <ToggleGroupItem
                            v-for="option in selectedOptions"
                            :key="option.value"
                            :value="option.value"
                        >
                            {{ option.label }}
                        </ToggleGroupItem>
                    </ToggleGroup>
                    <p class="mt-1 text-xs text-muted-foreground">
                        Each selected mode runs and appears in the comparison. This one decides
                        which mode's RERAs mark recovery breaths in the saved per-breath data, which
                        the experimental RERA estimate and breath tables use.
                    </p>
                </div>
            </CollapsibleContent>
        </Collapsible>
    </div>
</template>

<script setup lang="ts">
// Detection-mode selection shared by the single-session and batch analysis
// runs: any non-empty subset of modes, plus the primary mode the server
// requires to be one of them. The primary only picks which mode's RERAs flag
// recovery breaths in the stored per-breath rows, so it sits under Advanced.
import { computed, ref, useId, watch } from 'vue'
import { ChevronRight } from '@lucide/vue'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import type { AnalysisMode } from '@/api/analysis'

// Canonical order; AASM first, so the first selected mode is also the
// server's default (AASM when selected).
const MODE_OPTIONS: { value: AnalysisMode; label: string }[] = [
    { value: 'aasm', label: 'AASM' },
    { value: 'aasm_relaxed', label: 'AASM Relaxed' },
    { value: 'resmed', label: 'ResMed' },
]

const modes = defineModel<AnalysisMode[]>('modes', { required: true })
const primaryMode = defineModel<AnalysisMode>('primaryMode', { required: true })

const modesLabelId = useId()
const primaryLabelId = useId()

const selectedOptions = computed(() => MODE_OPTIONS.filter((o) => modes.value.includes(o.value)))

// The user's explicit Advanced pick; null while the primary follows the default.
// A selected non-default primary passed in on mount is an earlier pick (the run
// dialog remounts the picker each time it opens), so it is kept.
const userPick = ref<AnalysisMode | null>(
    modes.value.includes(primaryMode.value) && primaryMode.value !== selectedOptions.value[0]?.value
        ? primaryMode.value
        : null,
)

function modeLabel(mode: AnalysisMode): string {
    return MODE_OPTIONS.find((o) => o.value === mode)?.label ?? mode
}

// Keep the selection in canonical order rather than click order.
function setModes(value: unknown): void {
    const selected = value as AnalysisMode[]
    modes.value = MODE_OPTIONS.map((o) => o.value).filter((m) => selected.includes(m))
}

function pickPrimary(value: unknown): void {
    if (!value) return
    userPick.value = value as AnalysisMode
    primaryMode.value = userPick.value
}

// The primary is the user's pick while it stays selected, otherwise the first
// selected mode. An empty selection keeps the last primary, since nothing can
// run until one is chosen.
watch(
    modes,
    (selected) => {
        if (userPick.value && !selected.includes(userPick.value)) userPick.value = null
        const next = userPick.value ?? selectedOptions.value[0]?.value
        if (next && next !== primaryMode.value) primaryMode.value = next
    },
    { immediate: true },
)
</script>

<style scoped>
.mode-picker {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

@media (max-width: 767.98px) {
    .picker-toggles {
        flex-wrap: wrap;
    }

    /* shadcn size utilities are layered and lose to scoped CSS, so touch sizing lives here */
    .picker-toggles :deep(button),
    .advanced-trigger {
        min-height: var(--tap-target);
    }
}
</style>
