<template>
    <section v-if="rows.length" class="section-card">
        <h2>Flow Limitation &amp; Snore</h2>
        <Table>
            <TableHeader>
                <TableRow>
                    <TableHead>Channel</TableHead>
                    <TableHead>Median</TableHead>
                    <TableHead>95th</TableHead>
                    <TableHead>Max</TableHead>
                </TableRow>
            </TableHeader>
            <TableBody>
                <TableRow v-for="row in rows" :key="row.glossaryKey">
                    <TableCell class="whitespace-normal">
                        {{ GLOSSARY[row.glossaryKey].label }}
                        <InfoHint :glossary-key="row.glossaryKey" />
                    </TableCell>
                    <TableCell v-for="(value, i) in row.values" :key="i">
                        {{ value?.toFixed(2) ?? '---' }}
                    </TableCell>
                </TableRow>
            </TableBody>
        </Table>
    </section>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table'
import InfoHint from '@/components/InfoHint.vue'
import { GLOSSARY } from '@/utils/glossary'
import type { DeviceChannelStats } from '@/types'

const props = defineProps<{ stats?: DeviceChannelStats | null }>()

const rows = computed(() => {
    const s = props.stats
    if (!s) return []
    return [
        {
            glossaryKey: 'fl_device',
            values: [s.device_flg_median, s.device_flg_95th, s.device_flg_max],
        },
        { glossaryKey: 'snore_device', values: [s.snore_median, s.snore_95th, s.snore_max] },
    ].filter((row) => row.values.some((value) => value != null))
})
</script>
