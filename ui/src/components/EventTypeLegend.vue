<template>
    <ul class="space-y-1.5">
        <li v-for="code in displayTypes" :key="code" class="flex items-start gap-2 text-xs">
            <span
                class="mt-0.5 h-3 w-3 shrink-0 rounded-sm border border-border"
                :style="{ backgroundColor: EVENT_COLORS[code] }"
            />
            <span>
                <span class="font-medium">{{
                    GLOSSARY['event_type_' + code.toLowerCase()]?.label ?? code
                }}</span>
                <span class="text-muted-foreground">
                    — {{ GLOSSARY['event_type_' + code.toLowerCase()]?.short ?? '' }}</span
                >
            </span>
        </li>
    </ul>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { EVENT_COLORS } from '@/types'
import { GLOSSARY } from '@/utils/glossary'

const ALL_TYPES = ['OA', 'CA', 'UA', 'MA', 'H', 'RE', 'FL'] as const
type EventTypeCode = (typeof ALL_TYPES)[number]

const props = defineProps<{
    types?: EventTypeCode[]
}>()

const displayTypes = computed(() => props.types ?? [...ALL_TYPES])
</script>
