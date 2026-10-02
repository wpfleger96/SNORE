<template>
    <div
        role="note"
        :class="['flex items-start gap-2 rounded-md border px-3 py-2 text-sm', style.toneClass]"
    >
        <component :is="style.icon" class="mt-0.5 h-4 w-4 flex-shrink-0" aria-hidden="true" />
        <p>
            <!-- The space keeps title and body apart; Vue drops whitespace between tags. -->
            <span class="font-medium">{{ title }}</span
            >{{ ' ' }}<slot>{{ body }}</slot>
        </p>
    </div>
</template>

<script setup lang="ts">
// Section-level notice that the content below is an experimental SNORE
// heuristic. Pass `body` (or the default slot) to replace the standard text.
import { PROVENANCE_MARK_STYLES } from '@/utils/provenance'

withDefaults(
    defineProps<{
        title?: string
        body?: string
    }>(),
    {
        title: 'Experimental metric.',
        body: 'These are internally-consistent trend instruments, not clinically validated absolute measurements. Read them for night-to-night direction, not ground truth.',
    },
)

const style = PROVENANCE_MARK_STYLES.experimental
</script>
