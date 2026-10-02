<template>
    <div class="equipment-view">
        <h1 class="page-title">Equipment</h1>

        <!-- Devices section -->
        <h2 class="section-heading">Devices</h2>

        <div v-if="loading" class="loading-state">
            <Loader2 class="inline h-4 w-4 animate-spin" /> Loading devices...
        </div>

        <ErrorState v-else-if="error" :message="error" :retry="reload" />

        <div v-else-if="!devices.length" class="empty-state">
            <HardDrive class="h-10 w-10 text-muted-foreground mb-3" />
            <p class="text-muted-foreground">Import data to see devices.</p>
        </div>

        <div v-else class="devices-list">
            <div v-for="device in devices" :key="device.id" class="section-card device-card">
                <!-- Device header -->
                <div class="device-header">
                    <div>
                        <h2 class="device-title">{{ device.manufacturer }} {{ device.model }}</h2>
                        <span class="text-sm text-muted-foreground">
                            S/N: {{ device.serial_number }}
                        </span>
                    </div>
                </div>

                <!-- Identity grid -->
                <div class="identity-grid">
                    <div class="identity-row">
                        <span class="identity-label">Firmware</span>
                        <span class="identity-value">{{ device.firmware_version ?? '—' }}</span>
                    </div>
                    <div class="identity-row">
                        <span class="identity-label">Hardware</span>
                        <span class="identity-value">{{ device.hardware_version ?? '—' }}</span>
                    </div>
                    <div class="identity-row">
                        <span class="identity-label">Product Code</span>
                        <span class="identity-value">{{ device.product_code ?? '—' }}</span>
                    </div>
                    <div class="identity-row">
                        <span class="identity-label">First Seen</span>
                        <span class="identity-value">{{ formatDateFull(device.first_seen) }}</span>
                    </div>
                    <div class="identity-row">
                        <span class="identity-label">Last Import</span>
                        <span class="identity-value">{{
                            device.last_import ? formatDateFull(device.last_import) : '—'
                        }}</span>
                    </div>
                </div>

                <!-- Usage row -->
                <div class="usage-row">
                    <div class="usage-stat">
                        <span class="usage-value">{{ device.usage.session_count }}</span>
                        <span class="usage-label">Sessions</span>
                    </div>
                    <div class="usage-stat">
                        <span class="usage-value">{{
                            device.usage.total_therapy_hours.toFixed(1)
                        }}</span>
                        <span class="usage-label">Total Hours</span>
                    </div>
                    <div class="usage-stat">
                        <span class="usage-value">{{
                            device.usage.first_session_date
                                ? formatDateFull(device.usage.first_session_date)
                                : '—'
                        }}</span>
                        <span class="usage-label">First Session</span>
                    </div>
                    <div class="usage-stat">
                        <span class="usage-value">{{
                            device.usage.last_session_date
                                ? formatDateFull(device.usage.last_session_date)
                                : '—'
                        }}</span>
                        <span class="usage-label">Last Session</span>
                    </div>
                    <div v-if="device.usage.therapy_modes.length" class="usage-stat">
                        <span class="usage-value">{{ device.usage.therapy_modes.join(', ') }}</span>
                        <span class="usage-label">Modes</span>
                    </div>
                </div>

                <!-- Current Settings -->
                <Collapsible v-model:open="settingsOpen[device.id]" class="stats-collapsible">
                    <CollapsibleTrigger as-child>
                        <button class="collapsible-header">
                            Current Settings
                            <ChevronDown
                                class="h-4 w-4 transition-transform"
                                :class="{ 'rotate-180': settingsOpen[device.id] }"
                            />
                        </button>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                        <div
                            v-if="!categorizedSettings.get(device.id)?.length"
                            class="text-sm text-muted-foreground py-2"
                        >
                            No settings recorded for this device.
                        </div>
                        <div v-else class="settings-groups">
                            <div
                                v-for="group in categorizedSettings.get(device.id)"
                                :key="group.label"
                                class="settings-group"
                            >
                                <h4 class="settings-group-label">{{ group.label }}</h4>
                                <dl class="settings-list">
                                    <div
                                        v-for="entry in group.entries"
                                        :key="entry.key"
                                        class="setting-row"
                                    >
                                        <dt class="setting-key">
                                            {{ entry.label }}
                                            <InfoHint
                                                v-if="entry.glossaryKey"
                                                :glossary-key="entry.glossaryKey"
                                            />
                                        </dt>
                                        <dd class="setting-val">{{ entry.value }}</dd>
                                    </div>
                                </dl>
                            </div>
                        </div>
                    </CollapsibleContent>
                </Collapsible>

                <!-- Settings History -->
                <Collapsible v-model:open="historyOpen[device.id]" class="stats-collapsible">
                    <CollapsibleTrigger as-child>
                        <button class="collapsible-header">
                            Settings History
                            <ChevronDown
                                class="h-4 w-4 transition-transform"
                                :class="{ 'rotate-180': historyOpen[device.id] }"
                            />
                        </button>
                    </CollapsibleTrigger>
                    <CollapsibleContent>
                        <div
                            v-if="!device.settings_history.length"
                            class="text-sm text-muted-foreground py-2"
                        >
                            No settings changes detected.
                        </div>
                        <div v-else class="history-list">
                            <div
                                v-for="change in device.settings_history"
                                :key="change.session_id"
                                class="history-entry"
                            >
                                <span class="history-date">{{ formatDateFull(change.date) }}</span>
                                <ul class="history-changes">
                                    <li
                                        v-for="c in change.changes"
                                        :key="c.key"
                                        class="history-change"
                                    >
                                        <span class="history-key">{{ settingLabel(c.key) }}</span
                                        >:
                                        <template v-if="c.old_value === null">
                                            <span class="history-val new">
                                                {{ formatSettingValue(c.key, c.new_value ?? '') }}
                                                <em>(new)</em>
                                            </span>
                                        </template>
                                        <template v-else-if="c.new_value === null">
                                            <span class="history-val removed">
                                                {{ formatSettingValue(c.key, c.old_value ?? '') }}
                                                <em>(removed)</em>
                                            </span>
                                        </template>
                                        <template v-else>
                                            <span class="history-val old">{{
                                                formatSettingValue(c.key, c.old_value)
                                            }}</span>
                                            <span class="history-arrow">→</span>
                                            <span class="history-val new">{{
                                                formatSettingValue(c.key, c.new_value)
                                            }}</span>
                                        </template>
                                    </li>
                                </ul>
                            </div>
                        </div>
                    </CollapsibleContent>
                </Collapsible>
            </div>
        </div>

        <!-- Masks section -->
        <h2 class="section-heading">Masks</h2>
        <MaskLogManager :epochs="epochs" />
    </div>
</template>

<script setup lang="ts">
import { computed, reactive, watch } from 'vue'
import { ChevronDown, HardDrive, Loader2 } from '@lucide/vue'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import ErrorState from '@/components/ErrorState.vue'
import InfoHint from '@/components/InfoHint.vue'
import MaskLogManager from '@/components/MaskLogManager.vue'
import { getDevices, getDeviceDetail } from '@/api/devices'
import { getMaskEpochs } from '@/api/equipment'
import { useApiLoad } from '@/composables/useApiLoad'
import { formatDateFull } from '@/utils/formatting'
import { categorizeSettings, formatSettingValue, settingLabel } from '@/utils/deviceSettings'
import type { DeviceDetail, MaskEpochResponse } from '@/types'

const { data, loading, error, reload } = useApiLoad<DeviceDetail[]>(async () => {
    const list = await getDevices()
    return Promise.all(list.map((d) => getDeviceDetail(d.id)))
}, 'Failed to load devices')

const devices = computed(() => data.value ?? [])

// Epochs fetch: failure is silently degraded — MaskLogManager still works without epochs.
const { data: epochsData } = useApiLoad<MaskEpochResponse[]>(
    () => getMaskEpochs(),
    'Failed to load mask epochs',
)
const epochs = computed(() => epochsData.value ?? [])

// Default current-settings open, history collapsed per device
const settingsOpen = reactive<Record<number, boolean>>({})
const historyOpen = reactive<Record<number, boolean>>({})

// Computed once per devices change instead of three times per render per device.
const categorizedSettings = computed(
    () =>
        new Map(
            devices.value.map((d) => [
                d.id,
                d.current_settings ? categorizeSettings(d.current_settings) : [],
            ]),
        ),
)

watch(devices, (devs) => {
    for (const d of devs) {
        if (!(d.id in settingsOpen)) settingsOpen[d.id] = true
        if (!(d.id in historyOpen)) historyOpen[d.id] = false
    }
})
</script>

<style scoped>
.equipment-view {
    max-width: 900px;
}

.section-heading {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--color-muted-foreground);
    margin: 1.5rem 0 0.75rem;
}

.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    padding: 4rem 2rem;
    color: var(--color-muted-foreground);
}

.devices-list {
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

.device-card {
    padding: 1.25rem 1.5rem;
}

.device-header {
    margin-bottom: 1rem;
}

.device-title {
    font-size: 1.1rem;
    font-weight: 600;
    margin: 0 0 0.2rem;
}

.identity-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 0.5rem 1.5rem;
    margin-bottom: 1.25rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--color-border);
}

.identity-row {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
}

.identity-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--color-muted-foreground);
}

.identity-value {
    font-size: 0.875rem;
}

.usage-row {
    display: flex;
    flex-wrap: wrap;
    gap: 1rem 2rem;
    margin-bottom: 1.25rem;
    padding-bottom: 1.25rem;
    border-bottom: 1px solid var(--color-border);
}

.usage-stat {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
}

.usage-value {
    font-size: 0.925rem;
    font-weight: 500;
}

.usage-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--color-muted-foreground);
}

.settings-groups {
    display: flex;
    flex-wrap: wrap;
    gap: 1.5rem;
    padding: 0.75rem 0;
}

.settings-group {
    min-width: 180px;
}

.settings-group-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--color-muted-foreground);
    margin: 0 0 0.5rem;
}

.settings-list {
    margin: 0;
    display: flex;
    flex-direction: column;
    gap: 0.3rem;
}

.setting-row {
    display: flex;
    gap: 0.5rem;
    font-size: 0.875rem;
}

.setting-key {
    display: inline-flex;
    align-items: center;
    gap: 0.25rem;
    color: var(--color-muted-foreground);
    min-width: 130px;
}

.setting-val {
    font-weight: 500;
}

.history-list {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
    padding: 0.75rem 0;
}

.history-entry {
    font-size: 0.875rem;
}

.history-date {
    font-weight: 600;
    display: block;
    margin-bottom: 0.3rem;
}

.history-changes {
    margin: 0;
    padding-left: 1rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    list-style: none;
}

.history-change {
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 0.3rem;
}

.history-key {
    color: var(--color-muted-foreground);
}

.history-arrow {
    color: var(--color-muted-foreground);
}

.history-val {
    font-weight: 500;
}

.history-val.old {
    text-decoration: line-through;
    opacity: 0.6;
}

.history-val em {
    font-weight: 400;
    font-style: italic;
    color: var(--color-muted-foreground);
    font-size: 0.8em;
}

@media (max-width: 767.98px) {
    .device-card {
        padding: 0.875rem 1rem;
    }

    .settings-groups {
        flex-direction: column;
        gap: 1rem;
    }

    .settings-group {
        min-width: 0;
    }

    .setting-row {
        justify-content: space-between;
    }

    .setting-key {
        min-width: 0;
    }

    .setting-val {
        text-align: right;
    }
}
</style>
