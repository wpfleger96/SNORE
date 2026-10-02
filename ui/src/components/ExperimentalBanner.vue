<template>
    <div
        role="note"
        class="flex items-start gap-2 rounded-md border border-amber-300 bg-amber-50 px-3 py-2 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950/40 dark:text-amber-200"
    >
        <FlaskConical class="mt-0.5 h-4 w-4 flex-shrink-0" aria-hidden="true" />
        <p>
            <span class="font-medium">{{ title }}</span> {{ body }}
        </p>
    </div>
</template>

<script setup lang="ts">
// Section-level notice that the content below is SNORE's own heuristic
// (an experimental metric), not device data.
import { FlaskConical } from '@lucide/vue'

withDefaults(defineProps<{ title?: string; body?: string }>(), {
    title: 'Experimental metric.',
    body: 'These are internally-consistent trend instruments, not clinically validated absolute measurements. Read them for night-to-night direction, not ground truth.',
})
</script>
