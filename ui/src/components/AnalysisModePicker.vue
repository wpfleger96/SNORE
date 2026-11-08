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
        </div>
        <div v-if="modes.length > 1" class="primary-mode">
            <span :id="primaryLabelId" class="text-sm font-medium">Primary mode</span>
            <ToggleGroup
                type="single"
                variant="outline"
                class="picker-toggles mt-1"
                :aria-labelledby="primaryLabelId"
                :model-value="primaryMode"
                @update:model-value="
                    (v) => {
                        if (v) primaryMode = v as AnalysisMode
                    }
                "
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
                The primary mode's recovery markers are stored with the results.
            </p>
        </div>
    </div>
</template>

<script setup lang="ts">
// Detection-mode selection shared by the single-session and batch analysis
// runs: any non-empty subset of modes, plus the primary mode the server
// requires to be one of them.
import { computed, useId, watch } from 'vue'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import type { AnalysisMode } from '@/api/analysis'

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

// Keep the selection in canonical order rather than click order.
function setModes(value: unknown): void {
    const selected = value as AnalysisMode[]
    modes.value = MODE_OPTIONS.map((o) => o.value).filter((m) => selected.includes(m))
}

// A deselected primary falls back to the first mode still selected; an empty
// selection keeps the last primary, since nothing can run until one is chosen.
watch(
    modes,
    (selected) => {
        if (selected.length && !selected.includes(primaryMode.value)) {
            primaryMode.value = selected[0]!
        }
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
    .picker-toggles :deep(button) {
        min-height: var(--tap-target);
    }
}
</style>
