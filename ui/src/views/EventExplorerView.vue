<template>
    <div v-if="loading" class="loading-state">
        <Loader2 class="inline h-4 w-4 animate-spin" /> Loading events...
    </div>

    <ErrorState v-else-if="error" :message="error" :retry="reload" />

    <div v-else class="event-explorer">
        <RouterLink :to="{ name: 'session-detail', params: { id: sessionId } }" class="back-link">
            <ArrowLeft class="inline h-4 w-4" /> Back to Session
        </RouterLink>

        <h1 class="page-title">Events — Session #{{ sessionId }}</h1>

        <!-- Summary -->
        <div class="summary-row">
            <StatCard label="Total Events" :value="filteredEvents.length" :decimals="0" />
            <StatCard label="Events/Hour" :value="eventsPerHour" :decimals="1" />
            <StatCard label="Types" :value="uniqueTypes.length" :decimals="0" />
        </div>

        <div class="section-card">
            <h2>Device-scored Events</h2>

            <!-- Filters -->
            <div class="filter-bar">
                <span class="filter-label text-muted-foreground">Filter by type:</span>
                <div class="type-chips">
                    <button
                        v-for="t in uniqueTypes"
                        :key="t"
                        class="inline-flex cursor-pointer items-center rounded-full border px-2.5 py-0.5 text-xs font-semibold transition-colors"
                        :class="
                            activeTypes.has(t)
                                ? 'bg-primary text-primary-foreground'
                                : 'bg-muted/50 text-muted-foreground opacity-50'
                        "
                        @click="toggleType(t)"
                    >
                        {{ t }}
                    </button>
                </div>
                <InfoHint label="Event Types">
                    <EventTypeLegend />
                </InfoHint>
            </div>

            <!-- Event List (mobile cards) -->
            <div v-if="isMobile" class="card-list">
                <div v-for="(row, i) in paginatedEvents" :key="i" class="data-card">
                    <div class="data-card-header">
                        <span
                            class="event-badge"
                            :style="{ background: EVENT_COLORS[row.event_type] ?? '#ccc' }"
                        >
                            {{ row.event_type }}
                        </span>
                        <button
                            type="button"
                            class="time-link text-primary"
                            @click="jumpToWaveform(row.offset_seconds)"
                        >
                            {{ formatTimeOffset(row.offset_seconds) }}
                        </button>
                    </div>
                    <div class="data-card-row">
                        <span class="data-card-label">Duration</span>
                        <span class="data-card-value">{{
                            formatDuration(row.duration_seconds)
                        }}</span>
                    </div>
                </div>
            </div>

            <!-- Event List (desktop table) -->
            <Table v-else>
                <TableHeader>
                    <TableRow>
                        <TableHead style="width: 80px">Type</TableHead>
                        <TableHead>Time</TableHead>
                        <TableHead style="width: 100px">Duration</TableHead>
                    </TableRow>
                </TableHeader>
                <TableBody>
                    <TableRow v-for="(row, i) in paginatedEvents" :key="i" class="odd:bg-muted/50">
                        <TableCell>
                            <span
                                class="event-badge"
                                :style="{ background: EVENT_COLORS[row.event_type] ?? '#ccc' }"
                            >
                                {{ row.event_type }}
                            </span>
                        </TableCell>
                        <TableCell>
                            <button
                                class="time-link text-primary"
                                @click="jumpToWaveform(row.offset_seconds)"
                            >
                                {{ formatTimeOffset(row.offset_seconds) }}
                            </button>
                        </TableCell>
                        <TableCell>{{ formatDuration(row.duration_seconds) }}</TableCell>
                    </TableRow>
                </TableBody>
            </Table>

            <PaginationBar v-bind="pagerProps" @page="setPageOffset" />
        </div>
    </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { useRouter } from 'vue-router'
import {
    Table,
    TableBody,
    TableCell,
    TableHead,
    TableHeader,
    TableRow,
} from '@/components/ui/table'
import { Loader2, ArrowLeft } from '@lucide/vue'
import StatCard from '@/components/StatCard.vue'
import PaginationBar from '@/components/PaginationBar.vue'
import InfoHint from '@/components/InfoHint.vue'
import EventTypeLegend from '@/components/EventTypeLegend.vue'
import { getSessionEvents } from '@/api/events'
import { getSession } from '@/api/sessions'
import { useApiLoad } from '@/composables/useApiLoad'
import { usePagedRows } from '@/composables/usePagedRows'
import { useIsMobile } from '@/composables/useIsMobile'
import { formatTimeOffset } from '@/utils/formatting'
import { EVENT_COLORS } from '@/types'
import type { EventItem } from '@/types'
import ErrorState from '@/components/ErrorState.vue'

const emDash = '\u2014' // em-dash used for null display
const props = defineProps<{ sessionId: number }>()
const router = useRouter()
const { isMobile } = useIsMobile()

const activeTypes = ref<Set<string>>(new Set())

const { data, loading, error, reload } = useApiLoad(async () => {
    const [events, session] = await Promise.all([
        getSessionEvents(props.sessionId),
        getSession(props.sessionId, false),
    ])
    return { events, duration: session.duration_hours }
}, 'Failed to load events')

const allEvents = computed<EventItem[]>(() => data.value?.events ?? [])
const sessionDuration = computed(() => data.value?.duration ?? 0)

const uniqueTypes = computed(() => [...new Set(allEvents.value.map((e) => e.event_type))].sort())

const filteredEvents = computed(() => {
    if (activeTypes.value.size === 0) return allEvents.value
    return allEvents.value.filter((e) => activeTypes.value.has(e.event_type))
})

const {
    pageRows: paginatedEvents,
    barProps: pagerProps,
    setOffset: setPageOffset,
} = usePagedRows(filteredEvents, 50)

const eventsPerHour = computed(() => {
    if (!sessionDuration.value) return null
    return filteredEvents.value.length / sessionDuration.value
})

function formatDuration(seconds: EventItem['duration_seconds']): string {
    return seconds != null ? `${seconds.toFixed(1)}s` : emDash
}

function toggleType(type: string): void {
    const s = new Set(activeTypes.value)
    if (s.has(type)) s.delete(type)
    else s.add(type)
    activeTypes.value = s
}

function jumpToWaveform(offsetSec: number): void {
    void router.push({
        name: 'session-detail',
        params: { id: props.sessionId },
        query: { t: String(Math.floor(offsetSec)) },
    })
}
</script>

<style scoped>
.event-explorer {
    max-width: 1100px;
}

.summary-row {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(150px, 1fr));
    gap: 0.75rem;
    margin-bottom: 1.25rem;
}

.type-chips {
    display: flex;
    gap: 0.4rem;
    flex-wrap: wrap;
}

.event-badge {
    display: inline-block;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    font-size: 0.75rem;
    font-weight: 600;
    color: var(--color-foreground);
}

.time-link {
    cursor: pointer;
    text-decoration: none;
    background: none;
    border: none;
    padding: 0;
    font: inherit;
    color: inherit;
}
.time-link:hover {
    text-decoration: underline;
}

@media (max-width: 767.98px) {
    .time-link {
        min-height: var(--tap-target);
    }
}
</style>
