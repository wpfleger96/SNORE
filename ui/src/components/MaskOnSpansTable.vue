<template>
    <section v-if="spans?.length" class="section-card">
        <h2>{{ title ?? 'Mask-On Spans' }}</h2>
        <p v-if="total != null && total > spans.length" class="mb-2 text-sm text-muted-foreground">
            Showing the {{ spans.length.toLocaleString() }} longest of {{ total.toLocaleString() }},
            in time order.
        </p>
        <div v-if="isMobile" class="card-list">
            <div v-for="span in spans" :key="spanKey(span)" class="data-card">
                <div class="data-card-row">
                    <span class="data-card-label">Start</span>
                    <span class="data-card-value">{{ formatDateTimeSeconds(span.start) }}</span>
                </div>
                <div class="data-card-row">
                    <span class="data-card-label">End</span>
                    <span class="data-card-value">{{ formatDateTimeSeconds(span.end) }}</span>
                </div>
                <div class="data-card-row">
                    <span class="data-card-label">Duration</span>
                    <span class="data-card-value">{{
                        formatTimeOffset(span.duration_seconds)
                    }}</span>
                </div>
            </div>
        </div>
        <Table v-else>
            <TableHeader>
                <TableRow>
                    <TableHead>Start</TableHead>
                    <TableHead>End</TableHead>
                    <TableHead>Duration</TableHead>
                </TableRow>
            </TableHeader>
            <TableBody>
                <TableRow v-for="span in spans" :key="spanKey(span)">
                    <TableCell>{{ formatDateTimeSeconds(span.start) }}</TableCell>
                    <TableCell>{{ formatDateTimeSeconds(span.end) }}</TableCell>
                    <TableCell>{{ formatTimeOffset(span.duration_seconds) }}</TableCell>
                </TableRow>
            </TableBody>
        </Table>
    </section>
</template>

<script setup lang="ts">
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table'
import { useIsMobile } from '@/composables/useIsMobile'
import { formatDateTimeSeconds, formatTimeOffset } from '@/utils/formatting'
import type { MaskOnSpan } from '@/types'

// Also lists Large Leak spans, which carry the same fields. `total` is the
// full count when the API lists only some of them.
defineProps<{ spans?: MaskOnSpan[] | null; title?: string; total?: number }>()

const { isMobile } = useIsMobile()

function spanKey(span: MaskOnSpan): string {
    return `${span.session_id}:${span.start}`
}
</script>
