<template>
    <section v-if="rows.length" class="section-card">
        <h2>Device Channel Statistics</h2>
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
                        {{ value?.toFixed(row.decimals) ?? '---' }}
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

// Each channel's fields are `<field>_median`, `<field>_95th` and `<field>_max`.
const CHANNELS = [
    { glossaryKey: 'fl_device', field: 'device_flg', decimals: 2 },
    { glossaryKey: 'snore_device', field: 'snore', decimals: 2 },
    { glossaryKey: 'resp_rate_device', field: 'respiratory_rate', decimals: 1 },
    { glossaryKey: 'tidal_volume_device', field: 'tidal_volume', decimals: 0 },
    { glossaryKey: 'mv_device', field: 'minute_ventilation', decimals: 1 },
    { glossaryKey: 'ti_device', field: 'ti', decimals: 2 },
    { glossaryKey: 'ie_ratio_device', field: 'ie_ratio', decimals: 1 },
    { glossaryKey: 'pulse_device', field: 'pulse', decimals: 0 },
] as const
const STATISTICS = ['median', '95th', 'max'] as const

const rows = computed(() => {
    const s = props.stats
    if (!s) return []
    return CHANNELS.map(({ glossaryKey, field, decimals }) => ({
        glossaryKey,
        decimals,
        values: STATISTICS.map((stat) => s[`${field}_${stat}` as const]),
    })).filter((row) => row.values.some((value) => value != null))
})
</script>
