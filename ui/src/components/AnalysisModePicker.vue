<template>
    <div class="mode-picker">
        <div>
            <span :id="modesLabelId" class="text-sm font-medium">Modes</span>
            <ToggleGroup
                type="multiple"
                variant="outline"
                class="picker-toggles toggle-strong mt-1"
                :aria-labelledby="modesLabelId"
                :model-value="modes"
                @update:model-value="setModes"
            >
                <ToggleGroupItem v-for="mode in ANALYSIS_MODES" :key="mode" :value="mode">
                    {{ MODE_LABELS[mode] }}
                </ToggleGroupItem>
            </ToggleGroup>
            <p v-if="!modes.length" class="mt-1 text-xs text-destructive">
                Select at least one mode.
            </p>
            <p class="mt-1 text-xs text-muted-foreground">
                Each selected mode runs separately and gets its own row in each session's Mode
                Comparison.
            </p>
        </div>
        <Collapsible v-if="showStoredRera && effective && modes.length > 1" v-slot="{ open }">
            <CollapsibleTrigger as-child>
                <button
                    type="button"
                    class="advanced-trigger flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground transition-colors"
                >
                    <ChevronRight
                        class="h-4 w-4 transition-transform"
                        :class="{ 'rotate-90': open }"
                    />
                    Advanced: RERA flags from {{ MODE_LABELS[effective] }}
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
                        class="picker-toggles toggle-strong mt-1"
                        :aria-labelledby="primaryLabelId"
                        :model-value="effective"
                        @update:model-value="pickPrimary"
                    >
                        <ToggleGroupItem v-for="mode in modes" :key="mode" :value="mode">
                            {{ MODE_LABELS[mode] }}
                        </ToggleGroupItem>
                    </ToggleGroup>
                    <p class="mt-1 text-xs text-muted-foreground">
                        Decides which mode's RERAs mark recovery breaths in the saved per-breath
                        data, used by the experimental RERA estimate and breath tables. Sessions
                        saved with different choices can't be compared in the experimental RERA
                        epoch comparison.
                    </p>
                </div>
            </CollapsibleContent>
        </Collapsible>
    </div>
</template>

<script setup lang="ts">
// Detection-mode selection shared by the single-session and batch analysis
// runs: any non-empty subset of modes, plus an optional explicit primary mode.
// The primary only picks which mode's RERAs flag recovery breaths in the stored
// per-breath rows, so it sits under Advanced. The parent owns the explicit pick
// (null follows the default) and sends effectivePrimaryMode() with the run.
import { computed, useId } from 'vue'
import { ChevronRight } from '@lucide/vue'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import type { AnalysisMode } from '@/api/analysis'
import { ANALYSIS_MODES, MODE_LABELS, effectivePrimaryMode } from '@/utils/analysisModes'

// Hide the primary choice when it cannot matter, e.g. for a run that saves nothing.
const { showStoredRera = true } = defineProps<{ showStoredRera?: boolean }>()

const modes = defineModel<AnalysisMode[]>('modes', { required: true })
const primaryMode = defineModel<AnalysisMode | null>('primaryMode', { required: true })

const modesLabelId = useId()
const primaryLabelId = useId()

const effective = computed(() => effectivePrimaryMode(modes.value, primaryMode.value))

// Keep the selection in canonical order rather than click order, and drop an
// explicit pick that is no longer selected so it cannot resurface later.
function setModes(value: unknown): void {
    const selected = ANALYSIS_MODES.filter((m) => (value as AnalysisMode[]).includes(m))
    modes.value = selected
    if (primaryMode.value && !selected.includes(primaryMode.value)) primaryMode.value = null
}

// Re-clicking the pressed item emits an empty value; it pins that mode instead
// of unsetting it, since the pressed item is always the effective primary.
function pickPrimary(value: unknown): void {
    primaryMode.value = (value as AnalysisMode | undefined) || effective.value || null
}
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
