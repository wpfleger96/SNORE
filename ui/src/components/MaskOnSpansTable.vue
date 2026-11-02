<template>
    <section v-if="spans?.length" class="section-card">
        <h2>Mask-On Spans</h2>
        <div v-if="isMobile" class="card-list">
            <div v-for="span in spans" :key="spanKey(span)" class="data-card">
                <div class="data-card-row">
                    <span class="data-card-label">Start</span>
                    <span class="data-card-value">{{ formatDateTime(span.start) }}</span>
                </div>
                <div class="data-card-row">
                    <span class="data-card-label">End</span>
                    <span class="data-card-value">{{ formatDateTime(span.end) }}</span>
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
                    <TableCell>{{ formatDateTime(span.start) }}</TableCell>
                    <TableCell>{{ formatDateTime(span.end) }}</TableCell>
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
import { formatDateTime, formatTimeOffset } from '@/utils/formatting'
import type { MaskOnSpan } from '@/types'

defineProps<{ spans?: MaskOnSpan[] | null }>()

const { isMobile } = useIsMobile()

function spanKey(span: MaskOnSpan): string {
    return `${span.session_id}:${span.start}`
}
</script>
