<template>
    <div class="data-card">
        <div class="session-card-header">
            <input
                type="checkbox"
                :checked="selected"
                class="cursor-pointer"
                @change="$emit('toggleSelect')"
            />
            <RouterLink
                :to="{ name: 'session-detail', params: { id: session.id } }"
                class="session-card-date text-primary no-underline hover:underline"
            >
                <span class="block font-medium">{{
                    formatDateWithWeekday(session.therapy_day)
                }}</span>
                <span class="block text-xs text-muted-foreground">{{
                    formatDateTime(session.start_time)
                }}</span>
            </RouterLink>
            <DropdownMenu>
                <DropdownMenuTrigger as-child>
                    <Button variant="ghost" size="icon" class="session-card-menu" title="Actions">
                        <EllipsisVertical class="h-4 w-4" />
                    </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                    <DropdownMenuItem v-if="canWrite" @click="$emit('toggleEnabled')">
                        <Ban v-if="session.enabled" class="mr-2 h-4 w-4" />
                        <Check v-else class="mr-2 h-4 w-4" />
                        {{ session.enabled ? 'Disable' : 'Enable' }}
                    </DropdownMenuItem>
                    <DropdownMenuItem @click="$emit('events')">
                        <BarChart3 class="mr-2 h-4 w-4" />
                        Events
                    </DropdownMenuItem>
                    <DropdownMenuItem
                        v-if="canWrite"
                        variant="destructive"
                        @click="$emit('delete')"
                    >
                        <Trash2 class="mr-2 h-4 w-4" />
                        Delete
                    </DropdownMenuItem>
                </DropdownMenuContent>
            </DropdownMenu>
        </div>
        <div class="data-card-row">
            <span class="data-card-label">Duration</span>
            <span class="data-card-value">{{ session.duration_hours.toFixed(1) }}h</span>
        </div>
        <div class="data-card-row">
            <span class="data-card-label">AHI</span>
            <span class="data-card-value" :class="ahiClass(session.ahi)">
                {{ session.ahi?.toFixed(1) ?? '---' }}
            </span>
        </div>
        <div class="data-card-row">
            <span class="data-card-label">Device</span>
            <span class="data-card-value">{{ session.manufacturer }} {{ session.model }}</span>
        </div>
        <div class="data-card-row">
            <span class="data-card-label">Status</span>
            <span class="data-card-value">
                <Badge
                    v-if="session.enabled"
                    class="bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200"
                >
                    Active
                </Badge>
                <Badge v-else variant="secondary">Disabled</Badge>
            </span>
        </div>
    </div>
</template>

<script setup lang="ts">
import { Ban, BarChart3, Check, EllipsisVertical, Trash2 } from '@lucide/vue'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { ahiClass, formatDateTime, formatDateWithWeekday } from '@/utils/formatting'
import type { SessionListItem } from '@/types'

defineProps<{
    session: SessionListItem
    selected: boolean
    canWrite: boolean
}>()

defineEmits<{
    toggleSelect: []
    toggleEnabled: []
    events: []
    delete: []
}>()
</script>

<style scoped>
.session-card-header {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.5rem;
}

.session-card-date {
    flex: 1;
    min-width: 0;
}

/* Sole route to row actions on mobile — enforce the 44px touch floor
   (scoped unlayered rule beats the layered size-9 utility) */
.session-card-menu {
    min-height: var(--tap-target);
    min-width: var(--tap-target);
}
</style>
