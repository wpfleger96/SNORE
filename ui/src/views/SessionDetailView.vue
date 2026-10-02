<template>
    <div v-if="loading" class="session-detail">
        <Skeleton class="h-4 w-24 mb-4" />
        <Skeleton class="h-8 w-72 mb-2" />
        <div class="flex gap-2 mb-6">
            <Skeleton class="h-5 w-16 rounded-full" />
            <Skeleton class="h-5 w-32" />
            <Skeleton class="h-5 w-20" />
            <Skeleton class="h-5 w-16" />
        </div>
        <Skeleton class="h-[280px] w-full rounded-lg mb-6" />
        <Skeleton class="h-6 w-24 mb-3" />
        <div class="grid grid-cols-2 gap-3 sm:grid-cols-4">
            <Skeleton v-for="i in 8" :key="i" class="h-[88px] rounded-lg" />
        </div>
    </div>

    <div v-else-if="error" class="error-state">
        <AlertTriangle class="inline h-4 w-4" /> {{ error }}
    </div>

    <div v-else-if="session" class="session-detail">
        <!-- Back link -->
        <RouterLink to="/sessions" class="back-link">
            <ArrowLeft class="inline h-4 w-4" /> All Sessions
        </RouterLink>

        <!-- Session header -->
        <div class="session-header">
            <div>
                <div class="session-title-nav">
                    <button
                        type="button"
                        class="nav-arrow"
                        :disabled="!prevDay"
                        aria-label="Previous night"
                        @click="prevDay && navigateToDate(prevDay)"
                    >
                        <ChevronLeft class="h-5 w-5" />
                    </button>

                    <Popover v-model:open="datePickerOpen">
                        <PopoverTrigger as-child>
                            <button type="button" class="session-title-trigger">
                                {{ formatDateWithWeekday(session.therapy_day) }}
                                <ChevronDown class="h-4 w-4 opacity-60" />
                            </button>
                        </PopoverTrigger>
                        <PopoverContent class="w-auto p-0" align="center">
                            <Calendar
                                :model-value="strToCalendarDate(session.therapy_day)"
                                layout="month-and-year"
                                :is-date-disabled="isDateDisabled"
                                :min-value="minValue"
                                :max-value="maxValue"
                                @update:model-value="onCalendarSelect"
                            />
                        </PopoverContent>
                    </Popover>

                    <button
                        type="button"
                        class="nav-arrow"
                        :disabled="!nextDay"
                        aria-label="Next night"
                        @click="nextDay && navigateToDate(nextDay)"
                    >
                        <ChevronRight class="h-5 w-5" />
                    </button>
                </div>
                <div class="session-meta text-muted-foreground">
                    <Badge v-if="session.therapy_mode">{{ session.therapy_mode }}</Badge>
                    <span>{{ session.device_manufacturer }} {{ session.device_model }}</span>
                    <span>{{ session.duration_hours.toFixed(1) }} hours</span>
                    <span>Started: {{ formatDateTime(session.start_time) }}</span>
                    <span v-if="session.statistics?.ahi != null">
                        AHI:
                        <strong :class="ahiClass(session.statistics.ahi)">{{
                            session.statistics.ahi.toFixed(1)
                        }}</strong>
                    </span>
                    <span>{{ session.event_count }} events</span>
                </div>
            </div>
        </div>

        <!-- Mask info -->
        <div
            v-if="session.active_mask || maskTypeFromSettings"
            class="mask-info text-sm text-muted-foreground"
        >
            <span v-if="session.active_mask">Mask: {{ maskInfoLine }}</span>
            <span v-if="session.active_mask && maskTypeFromSettings">·</span>
            <span v-if="maskTypeFromSettings">Device type: {{ maskTypeFromSettings }}</span>
        </div>

        <!-- Waveform section -->
        <div class="bg-card border border-border rounded-lg py-4 px-5 mb-6">
            <WaveformToolbar
                v-model="selectedType"
                :available-types="session.waveform_types"
                v-model:multi-waveform="multiMode"
                :chart-count="multiViewRef?.chartCount ?? 1"
                :can-zoom-in="canZoomIn"
                :can-zoom-out="canZoomOut"
                @reset-zoom="handleResetZoom"
                @zoom-in="handleZoomIn"
                @zoom-out="handleZoomOut"
                @add-chart="handleAddChart"
            />

            <template v-if="!multiMode">
                <!-- Full-height spinner only before first data; refetches keep the chart mounted with a corner spinner (:refetching) -->
                <div
                    v-if="waveformLoading && !waveformData"
                    class="h-60 flex items-center justify-center gap-2 text-muted-foreground"
                >
                    <Loader2 class="h-4 w-4 animate-spin" /> Loading waveform...
                </div>
                <div
                    v-else-if="waveformError && !waveformData"
                    class="h-60 flex items-center justify-center gap-2 text-destructive"
                >
                    {{ waveformError }}
                </div>
                <WaveformChart
                    v-else-if="waveformData"
                    ref="singleChartRef"
                    :timestamps="waveformData.timestamps"
                    :values="waveformData.values"
                    :unit="waveformData.unit"
                    :label="selectedType"
                    :waveform-type="selectedType"
                    :events="selectedType === 'flow' ? events : undefined"
                    :start-epoch="startEpoch"
                    :refetching="waveformLoading"
                    @zoom="handleZoom"
                />
                <div
                    v-if="waveformData && waveformError && !waveformLoading"
                    class="mt-1 text-sm text-destructive"
                >
                    Failed to refresh waveform: {{ waveformError }}
                </div>
            </template>

            <MultiWaveformView
                v-else
                ref="multiViewRef"
                :session-id="session.id"
                :available-types="session.waveform_types"
                :events="events"
                :initial-types="[selectedType]"
                :start-epoch="startEpoch"
                :cache-registry="cacheRegistry"
                :duration-sec="fullDuration"
                @zoom="handleZoom"
            />
        </div>

        <!-- Statistics -->
        <div v-if="session.statistics" class="stats-section">
            <h2>Statistics</h2>

            <Collapsible v-model:open="respiratoryOpen" class="stats-collapsible">
                <CollapsibleTrigger as-child>
                    <button class="collapsible-header">
                        Respiratory Events
                        <ChevronDown
                            class="h-4 w-4 transition-transform"
                            :class="{ 'rotate-180': respiratoryOpen }"
                        />
                    </button>
                </CollapsibleTrigger>
                <CollapsibleContent>
                    <div class="stats-grid">
                        <StatCard
                            label="AHI"
                            :value="session.statistics.ahi"
                            :decimals="1"
                            glossary-key="ahi"
                        />
                        <StatCard
                            label="REI"
                            :value="session.statistics.rei"
                            :decimals="1"
                            glossary-key="rei"
                        />
                        <StatCard
                            label="OAI"
                            :value="session.statistics.oai"
                            :decimals="2"
                            glossary-key="oai"
                        />
                        <StatCard
                            label="CAI"
                            :value="session.statistics.cai"
                            :decimals="2"
                            glossary-key="cai"
                        />
                        <StatCard
                            label="HI"
                            :value="session.statistics.hi"
                            :decimals="2"
                            glossary-key="hi"
                        />
                        <StatCard
                            label="Obstructive Apneas"
                            :value="session.statistics.obstructive_apneas"
                            :decimals="0"
                            glossary-key="obstructive_apneas"
                        />
                        <StatCard
                            label="Central Apneas"
                            :value="session.statistics.central_apneas"
                            :decimals="0"
                            glossary-key="central_apneas"
                        />
                        <StatCard
                            label="Mixed Apneas"
                            :value="session.statistics.mixed_apneas"
                            :decimals="0"
                            glossary-key="mixed_apneas"
                        />
                        <StatCard
                            label="Hypopneas"
                            :value="session.statistics.hypopneas"
                            :decimals="0"
                            glossary-key="hypopneas"
                        />
                        <StatCard
                            label="RERAs"
                            :value="session.statistics.reras"
                            :decimals="0"
                            glossary-key="reras"
                        />
                        <StatCard
                            label="Flow Limitations"
                            :value="session.statistics.flow_limitations"
                            :decimals="0"
                            glossary-key="flow_limitations"
                        />
                    </div>
                </CollapsibleContent>
            </Collapsible>

            <Collapsible v-model:open="pressureOpen" class="stats-collapsible">
                <CollapsibleTrigger as-child>
                    <button class="collapsible-header">
                        Pressure
                        <ChevronDown
                            class="h-4 w-4 transition-transform"
                            :class="{ 'rotate-180': pressureOpen }"
                        />
                    </button>
                </CollapsibleTrigger>
                <CollapsibleContent>
                    <div class="stats-grid">
                        <StatCard
                            label="Pressure Mean"
                            :value="session.statistics.pressure_mean"
                            unit="cmH₂O"
                            :decimals="1"
                            glossary-key="pressure"
                        />
                        <StatCard
                            label="Pressure Min"
                            :value="session.statistics.pressure_min"
                            unit="cmH₂O"
                            :decimals="1"
                        />
                        <StatCard
                            label="Pressure Max"
                            :value="session.statistics.pressure_max"
                            unit="cmH₂O"
                            :decimals="1"
                        />
                        <StatCard
                            label="Pressure Median"
                            :value="session.statistics.pressure_median"
                            unit="cmH₂O"
                            :decimals="1"
                        />
                        <StatCard
                            label="Pressure 95th"
                            :value="session.statistics.pressure_95th"
                            unit="cmH₂O"
                            :decimals="1"
                        />
                        <StatCard
                            label="EPAP Mean"
                            :value="session.statistics.epap_mean"
                            unit="cmH₂O"
                            :decimals="1"
                            glossary-key="epap"
                        />
                        <StatCard
                            label="EPAP Min"
                            :value="session.statistics.epap_min"
                            unit="cmH₂O"
                            :decimals="1"
                        />
                        <StatCard
                            label="EPAP Max"
                            :value="session.statistics.epap_max"
                            unit="cmH₂O"
                            :decimals="1"
                        />
                        <StatCard
                            label="EPAP Median"
                            :value="session.statistics.epap_median"
                            unit="cmH₂O"
                            :decimals="1"
                        />
                        <StatCard
                            label="EPAP 95th"
                            :value="session.statistics.epap_95th"
                            unit="cmH₂O"
                            :decimals="1"
                        />
                        <template
                            v-if="
                                session.statistics.ipap_median != null ||
                                session.statistics.ipap_95th != null ||
                                session.statistics.ipap_max != null
                            "
                        >
                            <StatCard
                                v-if="session.statistics.ipap_median != null"
                                label="IPAP Median"
                                :value="session.statistics.ipap_median"
                                unit="cmH₂O"
                                :decimals="1"
                                glossary-key="ipap"
                            />
                            <StatCard
                                v-if="session.statistics.ipap_95th != null"
                                label="IPAP 95th"
                                :value="session.statistics.ipap_95th"
                                unit="cmH₂O"
                                :decimals="1"
                            />
                            <StatCard
                                v-if="session.statistics.ipap_max != null"
                                label="IPAP Max"
                                :value="session.statistics.ipap_max"
                                unit="cmH₂O"
                                :decimals="1"
                            />
                        </template>
                    </div>
                </CollapsibleContent>
            </Collapsible>

            <Collapsible v-model:open="leakOpen" class="stats-collapsible">
                <CollapsibleTrigger as-child>
                    <button class="collapsible-header">
                        Leak
                        <ChevronDown
                            class="h-4 w-4 transition-transform"
                            :class="{ 'rotate-180': leakOpen }"
                        />
                    </button>
                </CollapsibleTrigger>
                <CollapsibleContent>
                    <div class="stats-grid">
                        <StatCard
                            label="Leak Mean"
                            :value="session.statistics.leak_mean"
                            unit="L/min"
                            :decimals="1"
                            glossary-key="leak"
                        />
                        <StatCard
                            label="Leak Min"
                            :value="session.statistics.leak_min"
                            unit="L/min"
                            :decimals="1"
                        />
                        <StatCard
                            label="Leak Max"
                            :value="session.statistics.leak_max"
                            unit="L/min"
                            :decimals="1"
                        />
                        <StatCard
                            label="Leak Median"
                            :value="session.statistics.leak_median"
                            unit="L/min"
                            :decimals="1"
                        />
                        <StatCard
                            label="Leak 70th"
                            :value="session.statistics.leak_percentile_70"
                            unit="L/min"
                            :decimals="1"
                        />
                        <StatCard
                            label="Leak 95th"
                            :value="session.statistics.leak_95th"
                            unit="L/min"
                            :decimals="1"
                        />
                    </div>
                </CollapsibleContent>
            </Collapsible>

            <Collapsible v-model:open="oximetryOpen" class="stats-collapsible">
                <CollapsibleTrigger as-child>
                    <button class="collapsible-header">
                        Oximetry
                        <ChevronDown
                            class="h-4 w-4 transition-transform"
                            :class="{ 'rotate-180': oximetryOpen }"
                        />
                    </button>
                </CollapsibleTrigger>
                <CollapsibleContent>
                    <div class="stats-grid">
                        <StatCard
                            label="SpO₂ Mean"
                            :value="session.statistics.spo2_mean"
                            unit="%"
                            :decimals="1"
                            glossary-key="spo2"
                        />
                        <StatCard
                            label="SpO₂ Min"
                            :value="session.statistics.spo2_min"
                            unit="%"
                            :decimals="1"
                        />
                        <StatCard
                            label="SpO₂ Max"
                            :value="session.statistics.spo2_max"
                            unit="%"
                            :decimals="1"
                        />
                        <StatCard
                            label="SpO₂ Median"
                            :value="session.statistics.spo2_median"
                            unit="%"
                            :decimals="1"
                        />
                        <StatCard
                            label="SpO₂ 95th"
                            :value="session.statistics.spo2_95th"
                            unit="%"
                            :decimals="1"
                        />
                        <StatCard
                            label="SpO₂ Below 90%"
                            :value="session.statistics.spo2_time_below_90"
                            unit="s"
                            :decimals="0"
                            glossary-key="spo2_below_90"
                        />
                        <StatCard
                            label="Pulse Mean"
                            :value="session.statistics.pulse_mean"
                            unit="bpm"
                            :decimals="0"
                            glossary-key="pulse"
                        />
                        <StatCard
                            label="Pulse Min"
                            :value="session.statistics.pulse_min"
                            unit="bpm"
                            :decimals="0"
                        />
                        <StatCard
                            label="Pulse Max"
                            :value="session.statistics.pulse_max"
                            unit="bpm"
                            :decimals="0"
                        />
                    </div>
                </CollapsibleContent>
            </Collapsible>

            <Collapsible v-model:open="ventilationOpen" class="stats-collapsible">
                <CollapsibleTrigger as-child>
                    <button class="collapsible-header">
                        Ventilation
                        <ChevronDown
                            class="h-4 w-4 transition-transform"
                            :class="{ 'rotate-180': ventilationOpen }"
                        />
                    </button>
                </CollapsibleTrigger>
                <CollapsibleContent>
                    <div class="stats-grid">
                        <StatCard
                            label="Resp Rate Mean"
                            :value="session.statistics.respiratory_rate_mean"
                            unit="br/min"
                            :decimals="1"
                            glossary-key="resp_rate"
                        />
                        <StatCard
                            label="Resp Rate Min"
                            :value="session.statistics.respiratory_rate_min"
                            unit="br/min"
                            :decimals="1"
                        />
                        <StatCard
                            label="Resp Rate Max"
                            :value="session.statistics.respiratory_rate_max"
                            unit="br/min"
                            :decimals="1"
                        />
                        <StatCard
                            v-if="session.statistics.respiratory_rate_95th != null"
                            label="Resp Rate 95th"
                            :value="session.statistics.respiratory_rate_95th"
                            unit="br/min"
                            :decimals="1"
                        />
                        <!-- STR tidal-volume stats are in L on the device; convert to mL for display. -->
                        <StatCard
                            label="Tidal Volume Mean"
                            :value="tvToMl(session.statistics.tidal_volume_mean)"
                            unit="mL"
                            :decimals="0"
                            glossary-key="tidal_volume"
                        />
                        <StatCard
                            label="Tidal Volume Min"
                            :value="tvToMl(session.statistics.tidal_volume_min)"
                            unit="mL"
                            :decimals="0"
                        />
                        <StatCard
                            label="Tidal Volume Max"
                            :value="tvToMl(session.statistics.tidal_volume_max)"
                            unit="mL"
                            :decimals="0"
                        />
                        <StatCard
                            v-if="session.statistics.tidal_volume_95th != null"
                            label="Tidal Volume 95th"
                            :value="tvToMl(session.statistics.tidal_volume_95th)"
                            unit="mL"
                            :decimals="0"
                        />
                        <StatCard
                            label="Min Ventilation Mean"
                            :value="session.statistics.minute_ventilation_mean"
                            unit="L/min"
                            :decimals="1"
                            glossary-key="mv"
                        />
                        <StatCard
                            label="Min Ventilation Min"
                            :value="session.statistics.minute_ventilation_min"
                            unit="L/min"
                            :decimals="1"
                        />
                        <StatCard
                            label="Min Ventilation Max"
                            :value="session.statistics.minute_ventilation_max"
                            unit="L/min"
                            :decimals="1"
                        />
                        <StatCard
                            v-if="session.statistics.minute_ventilation_95th != null"
                            label="Min Ventilation 95th"
                            :value="session.statistics.minute_ventilation_95th"
                            unit="L/min"
                            :decimals="1"
                        />
                    </div>
                </CollapsibleContent>
            </Collapsible>

            <!-- Device Indices (UAI, AI, RIN, CSR — device-conditional, may be NULL) -->
            <Collapsible
                v-if="hasDeviceIndices"
                v-model:open="deviceIndicesOpen"
                class="stats-collapsible"
            >
                <CollapsibleTrigger as-child>
                    <button class="collapsible-header">
                        Device Indices
                        <ChevronDown
                            class="h-4 w-4 transition-transform"
                            :class="{ 'rotate-180': deviceIndicesOpen }"
                        />
                    </button>
                </CollapsibleTrigger>
                <CollapsibleContent>
                    <div class="stats-grid">
                        <StatCard
                            v-if="session.statistics.uai != null"
                            label="UAI"
                            :value="session.statistics.uai"
                            unit="events/hr"
                            :decimals="2"
                            glossary-key="uai"
                        />
                        <StatCard
                            v-if="session.statistics.ai != null"
                            label="AI"
                            :value="session.statistics.ai"
                            unit="events/hr"
                            :decimals="2"
                            glossary-key="ai_str"
                        />
                        <StatCard
                            v-if="session.statistics.rin != null"
                            label="RIN"
                            :value="session.statistics.rin"
                            unit="events/hr"
                            :decimals="2"
                            glossary-key="rin"
                        />
                        <StatCard
                            v-if="session.statistics.csr_pct != null"
                            label="CSR"
                            :value="session.statistics.csr_pct"
                            unit="%"
                            :decimals="1"
                            glossary-key="csr"
                        />
                        <StatCard
                            v-if="session.statistics.spont_cyc_pct != null"
                            label="Spont Cyc"
                            :value="session.statistics.spont_cyc_pct"
                            unit="%"
                            :decimals="1"
                            glossary-key="spont_cyc_pct"
                        />
                        <StatCard
                            v-if="session.statistics.mask_events != null"
                            label="Mask Events"
                            :value="session.statistics.mask_events"
                            :decimals="0"
                            glossary-key="mask_events_str"
                        />
                    </div>
                </CollapsibleContent>
            </Collapsible>

            <!-- Flow & Pressure Percentiles (from STR blower-side signals) -->
            <Collapsible
                v-if="hasFlowPressure"
                v-model:open="flowPressureOpen"
                class="stats-collapsible"
            >
                <CollapsibleTrigger as-child>
                    <button class="collapsible-header">
                        Flow & Pressure Percentiles
                        <ChevronDown
                            class="h-4 w-4 transition-transform"
                            :class="{ 'rotate-180': flowPressureOpen }"
                        />
                    </button>
                </CollapsibleTrigger>
                <CollapsibleContent>
                    <div class="stats-grid">
                        <StatCard
                            v-if="session.statistics.flow_5th != null"
                            label="Flow 5th"
                            :value="session.statistics.flow_5th"
                            unit="L/min"
                            :decimals="1"
                            glossary-key="flow_5th"
                        />
                        <StatCard
                            v-if="session.statistics.flow_95th != null"
                            label="Flow 95th"
                            :value="session.statistics.flow_95th"
                            unit="L/min"
                            :decimals="1"
                        />
                        <StatCard
                            v-if="session.statistics.blow_press_5th != null"
                            label="Blow Press 5th"
                            :value="session.statistics.blow_press_5th"
                            unit="cmH₂O"
                            :decimals="1"
                            glossary-key="blow_press"
                        />
                        <StatCard
                            v-if="session.statistics.blow_press_95th != null"
                            label="Blow Press 95th"
                            :value="session.statistics.blow_press_95th"
                            unit="cmH₂O"
                            :decimals="1"
                        />
                        <StatCard
                            v-if="session.statistics.blow_flow_median != null"
                            label="Blow Flow Median"
                            :value="session.statistics.blow_flow_median"
                            unit="L/min"
                            :decimals="1"
                            glossary-key="blow_flow"
                        />
                    </div>
                </CollapsibleContent>
            </Collapsible>

            <!-- I:E Ratio & Inspiratory Time (VAuto only) -->
            <Collapsible v-if="hasIeTi" v-model:open="ieTiOpen" class="stats-collapsible">
                <CollapsibleTrigger as-child>
                    <button class="collapsible-header">
                        I:E Ratio & Ti
                        <ChevronDown
                            class="h-4 w-4 transition-transform"
                            :class="{ 'rotate-180': ieTiOpen }"
                        />
                    </button>
                </CollapsibleTrigger>
                <CollapsibleContent>
                    <div class="stats-grid">
                        <StatCard
                            v-if="session.statistics.ie_ratio_median != null"
                            label="I:E Ratio Median"
                            :value="session.statistics.ie_ratio_median"
                            unit="%"
                            :decimals="0"
                            glossary-key="ie_ratio_stat"
                        />
                        <StatCard
                            v-if="session.statistics.ie_ratio_95th != null"
                            label="I:E Ratio 95th"
                            :value="session.statistics.ie_ratio_95th"
                            unit="%"
                            :decimals="0"
                        />
                        <StatCard
                            v-if="session.statistics.ie_ratio_max != null"
                            label="I:E Ratio Max"
                            :value="session.statistics.ie_ratio_max"
                            unit="%"
                            :decimals="0"
                        />
                        <StatCard
                            v-if="session.statistics.ti_median != null"
                            label="Ti Median"
                            :value="session.statistics.ti_median"
                            unit="s"
                            :decimals="2"
                            glossary-key="ti_stat"
                        />
                        <StatCard
                            v-if="session.statistics.ti_95th != null"
                            label="Ti 95th"
                            :value="session.statistics.ti_95th"
                            unit="s"
                            :decimals="2"
                        />
                        <StatCard
                            v-if="session.statistics.ti_max != null"
                            label="Ti Max"
                            :value="session.statistics.ti_max"
                            unit="s"
                            :decimals="2"
                        />
                    </div>
                </CollapsibleContent>
            </Collapsible>

            <!-- Climate & Humidifier (present only when humidifier is active) -->
            <Collapsible v-if="hasClimate" v-model:open="climateOpen" class="stats-collapsible">
                <CollapsibleTrigger as-child>
                    <button class="collapsible-header">
                        Climate & Humidifier
                        <ChevronDown
                            class="h-4 w-4 transition-transform"
                            :class="{ 'rotate-180': climateOpen }"
                        />
                    </button>
                </CollapsibleTrigger>
                <CollapsibleContent>
                    <div class="stats-grid">
                        <StatCard
                            v-if="session.statistics.amb_humidity_median != null"
                            label="Amb Humidity Median"
                            :value="session.statistics.amb_humidity_median"
                            unit="%"
                            :decimals="1"
                            glossary-key="amb_humidity"
                        />
                        <StatCard
                            v-if="session.statistics.hum_temp_median != null"
                            label="Hum Temp Median"
                            :value="session.statistics.hum_temp_median"
                            unit="°C"
                            :decimals="1"
                            glossary-key="hum_temp"
                        />
                        <StatCard
                            v-if="session.statistics.htube_temp_median != null"
                            label="HTube Temp Median"
                            :value="session.statistics.htube_temp_median"
                            unit="°C"
                            :decimals="1"
                            glossary-key="htube_temp"
                        />
                        <StatCard
                            v-if="session.statistics.htube_pow_median != null"
                            label="HTube Power Median"
                            :value="session.statistics.htube_pow_median"
                            unit="W"
                            :decimals="1"
                            glossary-key="htube_pow"
                        />
                        <StatCard
                            v-if="session.statistics.hum_pow_median != null"
                            label="Hum Power Median"
                            :value="session.statistics.hum_pow_median"
                            unit="W"
                            :decimals="1"
                            glossary-key="hum_pow"
                        />
                    </div>
                </CollapsibleContent>
            </Collapsible>
        </div>

        <!-- Apple Health -->
        <Collapsible v-model:open="healthOpen" class="settings-panel">
            <CollapsibleTrigger as-child>
                <button
                    class="flex w-full items-center justify-between rounded-lg border border-border bg-card p-4 text-left font-semibold hover:bg-accent"
                >
                    Apple Health
                    <ChevronDown
                        class="h-4 w-4 transition-transform"
                        :class="{ 'rotate-180': healthOpen }"
                    />
                </button>
            </CollapsibleTrigger>
            <CollapsibleContent class="px-4 pt-3 pb-4">
                <div
                    v-if="healthLoading"
                    class="flex items-center gap-2 text-sm text-muted-foreground py-2"
                >
                    <Loader2 class="h-4 w-4 animate-spin" /> Loading...
                </div>
                <p v-else-if="healthNotFound" class="text-sm text-muted-foreground py-2">
                    No Apple Health data for this night.
                </p>
                <p v-else-if="healthError" class="text-sm text-destructive py-2">
                    {{ healthError }}
                </p>
                <template v-else-if="healthNight">
                    <div class="stats-grid mb-4">
                        <StatCard
                            label="Total Sleep"
                            :value="secToHours(healthNight.total_sleep_seconds)"
                            unit="hr"
                            :decimals="1"
                            glossary-key="total_sleep"
                        />
                        <StatCard
                            label="Efficiency"
                            :value="healthNight.sleep_efficiency_pct ?? null"
                            unit="%"
                            :decimals="1"
                            glossary-key="sleep_efficiency"
                        />
                        <StatCard
                            label="Deep"
                            :value="secToHours(healthNight.deep_seconds)"
                            unit="hr"
                            :decimals="1"
                            glossary-key="deep_sleep"
                        />
                        <StatCard
                            label="REM"
                            :value="secToHours(healthNight.rem_seconds)"
                            unit="hr"
                            :decimals="1"
                            glossary-key="rem_sleep"
                        />
                        <StatCard
                            v-if="healthNight.avg_spo2_pct != null"
                            label="SpO₂ Avg"
                            :value="healthNight.avg_spo2_pct"
                            unit="%"
                            :decimals="1"
                        />
                        <StatCard
                            v-if="healthNight.avg_rr != null"
                            label="Resp Rate Avg"
                            :value="healthNight.avg_rr"
                            unit="br/min"
                            :decimals="1"
                        />
                    </div>
                    <div class="flex gap-4 text-sm flex-wrap">
                        <RouterLink
                            :to="`/apple-health/${session.therapy_day}`"
                            class="session-link"
                        >
                            Apple Health night detail →
                        </RouterLink>
                        <RouterLink :to="`/days/${session.therapy_day}`" class="session-link">
                            View therapy day →
                        </RouterLink>
                    </div>
                </template>
            </CollapsibleContent>
        </Collapsible>

        <!-- Import Provenance -->
        <Collapsible
            v-if="
                session.import_source ||
                session.parser_version ||
                session.data_quality_notes?.length
            "
            v-model:open="provenanceOpen"
            class="settings-panel"
        >
            <CollapsibleTrigger as-child>
                <button
                    class="flex w-full items-center justify-between rounded-lg border border-border bg-card p-4 text-left font-semibold hover:bg-accent"
                >
                    Import Provenance
                    <ChevronDown
                        class="h-4 w-4 transition-transform"
                        :class="{ 'rotate-180': provenanceOpen }"
                    />
                </button>
            </CollapsibleTrigger>
            <CollapsibleContent class="px-4 pt-3 pb-4">
                <div class="settings-grid">
                    <div v-if="session.import_source" class="setting-row">
                        <span class="setting-key text-muted-foreground">Import Source</span>
                        <span class="setting-value">{{ session.import_source }}</span>
                    </div>
                    <div v-if="session.parser_version" class="setting-row">
                        <span class="setting-key text-muted-foreground">Parser Version</span>
                        <span class="setting-value">{{ session.parser_version }}</span>
                    </div>
                </div>
                <div v-if="session.data_quality_notes?.length" class="quality-notes">
                    <p class="quality-notes-label text-muted-foreground">Data Quality Notes</p>
                    <div class="quality-chips">
                        <span
                            v-for="note in session.data_quality_notes"
                            :key="note"
                            class="quality-chip"
                        >
                            {{ note }}
                        </span>
                    </div>
                </div>
            </CollapsibleContent>
        </Collapsible>

        <!-- Device settings -->
        <Collapsible
            v-if="session.settings?.length"
            v-model:open="settingsOpen"
            class="settings-panel"
        >
            <CollapsibleTrigger as-child>
                <button
                    class="flex w-full items-center justify-between rounded-lg border border-border bg-card p-4 text-left font-semibold hover:bg-accent"
                >
                    Device Settings
                    <ChevronDown
                        class="h-4 w-4 transition-transform"
                        :class="{ 'rotate-180': settingsOpen }"
                    />
                </button>
            </CollapsibleTrigger>
            <CollapsibleContent class="px-4 pt-3 pb-4">
                <div class="settings-grid">
                    <div v-for="s in session.settings" :key="s.key" class="setting-row">
                        <span class="setting-key text-muted-foreground">{{ s.key }}</span>
                        <span class="setting-value">{{ s.value ?? '---' }}</span>
                    </div>
                </div>
            </CollapsibleContent>
        </Collapsible>
    </div>
</template>

<script setup lang="ts">
import { ref, computed, watch, nextTick, toRef } from 'vue'
import { isAxiosError } from 'axios'
import { useRoute, useRouter } from 'vue-router'
import type { DateValue } from 'reka-ui'
import { Badge } from '@/components/ui/badge'
import { Calendar } from '@/components/ui/calendar'
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible'
import { Popover, PopoverContent, PopoverTrigger } from '@/components/ui/popover'
import { Skeleton } from '@/components/ui/skeleton'
import {
    Loader2,
    AlertTriangle,
    ArrowLeft,
    ChevronDown,
    ChevronLeft,
    ChevronRight,
} from '@lucide/vue'
import WaveformChart from '@/components/WaveformChart.vue'
import WaveformToolbar from '@/components/WaveformToolbar.vue'
import MultiWaveformView from '@/components/MultiWaveformView.vue'
import StatCard from '@/components/StatCard.vue'
import { getSession } from '@/api/sessions'
import { getSessionEvents } from '@/api/events'
import { getDay } from '@/api/days'
import { getHealthNight } from '@/api/health'
import { useWaveformWindow } from '@/composables/useWaveformWindow'
import { createWaveformCacheRegistry } from '@/utils/waveformCache'
import { zoomInWindow, zoomOutWindow } from '@/utils/zoomMath'
import { MIN_ZOOM_WINDOW_SEC } from '@/constants/waveform'
import {
    useAvailableDates,
    strToCalendarDate,
    calendarDateToStr,
    adjacentDates,
} from '@/composables/useAvailableDates'
import {
    ahiClass,
    formatDateWithWeekday,
    formatDateFull,
    formatDateTime,
    secToHours,
} from '@/utils/formatting'
import { maskEntryName, styleLabel } from '@/utils/maskOptions'
import type { SessionDetail, EventItem, HealthNightDetailRead } from '@/types'

const props = defineProps<{ sessionId: number }>()
const route = useRoute()

const session = ref<SessionDetail | null>(null)
const events = ref<EventItem[]>([])
const loading = ref(true)
const error = ref<string | null>(null)

const selectedType = ref('')
const multiMode = ref(false)
const currentZoomRange = ref<{ start: number; end: number } | null>(null)
const settingsOpen = ref(false)
const respiratoryOpen = ref(true)
const pressureOpen = ref(false)
const leakOpen = ref(false)
const oximetryOpen = ref(false)
const ventilationOpen = ref(false)
const deviceIndicesOpen = ref(false)
const flowPressureOpen = ref(false)
const ieTiOpen = ref(false)
const climateOpen = ref(false)
const provenanceOpen = ref(false)
const healthOpen = ref(false)
const healthLoading = ref(false)
const healthNotFound = ref(false)
const healthError = ref<string | null>(null)
const healthNight = ref<HealthNightDetailRead | null>(null)
const healthFetched = ref(false)

const maskTypeFromSettings = computed(
    () => session.value?.settings?.find((s) => s.key === 'mask_type')?.value ?? null,
)

const maskInfoLine = computed(() => {
    const m = session.value?.active_mask
    if (!m) return ''
    const hasName = !!(m.brand || m.model)
    let line = maskEntryName(m)
    if (m.size) line += `, size ${m.size}`
    const parens: string[] = []
    if (hasName && m.style) parens.push(styleLabel(m.style))
    if (m.start_date) parens.push(`since ${formatDateFull(m.start_date)}`)
    if (parens.length) line += ` (${parens.join(', ')})`
    return line
})

// STR tidal-volume stats are stored in L; convert to mL for display.
function tvToMl(val: number | null | undefined): number | null {
    return val != null ? val * 1000 : null
}

// Conditional section visibility — collapses that only appear when the device
// reported at least one of the enclosed fields for this session.
const hasDeviceIndices = computed(() => {
    const s = session.value?.statistics
    if (!s) return false
    return (
        s.uai != null ||
        s.ai != null ||
        s.rin != null ||
        s.csr_pct != null ||
        s.spont_cyc_pct != null ||
        s.mask_events != null
    )
})

const hasFlowPressure = computed(() => {
    const s = session.value?.statistics
    if (!s) return false
    return (
        s.flow_5th != null ||
        s.flow_95th != null ||
        s.blow_press_5th != null ||
        s.blow_press_95th != null ||
        s.blow_flow_median != null
    )
})

const hasIeTi = computed(() => {
    const s = session.value?.statistics
    if (!s) return false
    return (
        s.ie_ratio_median != null ||
        s.ie_ratio_95th != null ||
        s.ie_ratio_max != null ||
        s.ti_median != null ||
        s.ti_95th != null ||
        s.ti_max != null
    )
})

const hasClimate = computed(() => {
    const s = session.value?.statistics
    if (!s) return false
    return (
        s.amb_humidity_median != null ||
        s.hum_temp_median != null ||
        s.htube_temp_median != null ||
        s.htube_pow_median != null ||
        s.hum_pow_median != null
    )
})

const router = useRouter()
const {
    load: loadDates,
    loaded: datesLoaded,
    minValue,
    maxValue,
    isDateDisabled,
    sortedDates,
} = useAvailableDates()

const sessionIdRef = toRef(props, 'sessionId')

// Declared above the composable call: useWaveformWindow captures fullDuration, and
// const bindings would be in the temporal dead zone if defined further down.
const startEpoch = computed(() => {
    if (!session.value) return 0
    const ms = new Date(session.value.start_time).getTime()
    return Number.isFinite(ms) ? Math.floor(ms / 1000) : 0
})

const fullDuration = computed(() => (session.value ? session.value.duration_hours * 3600 : 0))

// One registry per mounted view; shared by the single-mode composable and every multi-mode
// chart, persists across type switches and mode toggles, cleared only on session change.
const cacheRegistry = createWaveformCacheRegistry()

const {
    data: waveformData,
    loading: waveformLoading,
    error: waveformError,
    loadWindow,
    reset: resetWaveformData,
} = useWaveformWindow(sessionIdRef, selectedType, cacheRegistry, fullDuration)

const singleChartRef = ref<InstanceType<typeof WaveformChart>>()
const multiViewRef = ref<InstanceType<typeof MultiWaveformView>>()

const rawT = route.query.t ? Number(route.query.t) : null
const jumpToTime = rawT != null && Number.isFinite(rawT) ? rawT : null

// Jump to timestamp from ?t= query param after first waveform load
if (jumpToTime != null) {
    const stopWatch = watch(waveformData, (data) => {
        if (data) {
            stopWatch()
            nextTick(() => {
                const padding = 300
                const start = Math.max(0, jumpToTime - padding)
                const end = Math.min(fullDuration.value, jumpToTime + padding)
                currentZoomRange.value = { start, end }
                singleChartRef.value?.setScaleX(start, end)
                // Load dense data for the jumped window (and warm its neighbors); the overview alone
                // would render the jump sparsely until the user next interacts.
                loadWindow(start, end)
            })
        }
    })
}

function handleZoom(startSec: number, endSec: number): void {
    currentZoomRange.value = { start: startSec, end: endSec }
    if (!multiMode.value) {
        loadWindow(startSec, endSec)
    }
}

async function handleResetZoom(): Promise<void> {
    currentZoomRange.value = null
    if (multiMode.value) {
        multiViewRef.value?.resetZoom()
    } else {
        loadWindow()
        // resetZoom scales to the data extent, correct only after the full-night data has landed;
        // loadWindow resolves the never-evicted overview synchronously but props flush async.
        await nextTick()
        singleChartRef.value?.resetZoom()
    }
}

function applyZoom(startSec: number, endSec: number): void {
    currentZoomRange.value = { start: startSec, end: endSec }
    if (multiMode.value) {
        multiViewRef.value?.zoomTo(startSec, endSec)
    } else {
        loadWindow(startSec, endSec)
        // Data swaps no longer rescale (WaveformChart pins the viewport), so a button zoom must
        // move the viewport explicitly and immediately.
        singleChartRef.value?.setScaleX(startSec, endSec)
    }
}

function handleZoomIn(): void {
    if (!session.value) return
    const duration = fullDuration.value
    const range = currentZoomRange.value ?? { start: 0, end: duration }
    if (range.end - range.start <= MIN_ZOOM_WINDOW_SEC) return

    const { start, end } = zoomInWindow(range.start, range.end, duration)
    applyZoom(start, end)
}

function handleZoomOut(): void {
    if (!session.value || !currentZoomRange.value) return
    const duration = fullDuration.value
    const range = currentZoomRange.value
    if ((range.end - range.start) * 2 >= duration) {
        void handleResetZoom()
        return
    }

    const { start, end } = zoomOutWindow(range.start, range.end, duration)
    applyZoom(start, end)
}

function handleAddChart(): void {
    if (!session.value || !multiViewRef.value) return
    const usedTypes = multiViewRef.value.chartTypes()
    const next = session.value.waveform_types.find((t) => !usedTypes.includes(t))
    if (next) multiViewRef.value.addChart(next)
}

const canZoomIn = computed(() => {
    if (!session.value || fullDuration.value <= MIN_ZOOM_WINDOW_SEC) return false
    const currentWindow = currentZoomRange.value
        ? currentZoomRange.value.end - currentZoomRange.value.start
        : fullDuration.value
    return currentWindow > MIN_ZOOM_WINDOW_SEC
})

const canZoomOut = computed(() => currentZoomRange.value !== null)

const datePickerOpen = ref(false)

const prevDay = computed<string | null>(() => {
    if (!datesLoaded.value || !session.value) return null
    return adjacentDates(sortedDates.value, session.value.therapy_day).prev
})

const nextDay = computed<string | null>(() => {
    if (!datesLoaded.value || !session.value) return null
    return adjacentDates(sortedDates.value, session.value.therapy_day).next
})

function onCalendarSelect(val: DateValue | undefined): void {
    if (!val) return
    datePickerOpen.value = false
    void navigateToDate(calendarDateToStr(val))
}

let navGeneration = 0

async function navigateToDate(date: string): Promise<void> {
    const gen = ++navGeneration
    try {
        const day = await getDay(date)
        if (gen !== navGeneration) return
        // session_ids is ordered by start_time on the backend (day_service.py), so [0] is the night's first/primary session.
        const id = day.session_ids?.[0]
        if (id != null) {
            void router.push({ name: 'session-detail', params: { id } })
        } else {
            // No sessions for this date — stay on current session intentionally.
            console.warn(`navigateToDate: ${date} has no sessions`)
        }
    } catch {
        // date resolution failed — stay on current session
    }
}

watch(selectedType, (newType) => {
    if (!multiMode.value && newType) {
        resetWaveformData()
        loadWindow()
    }
})

// Re-sync single-mode data (and clear any stale error) when leaving multi mode;
// multi-mode zooms update currentZoomRange but never refetch the single chart
watch(multiMode, (multi) => {
    if (!multi) {
        loadWindow(currentZoomRange.value?.start, currentZoomRange.value?.end)
    }
})

let loadGeneration = 0

// Lazy-load Apple Health night on first open
watch(healthOpen, (isOpen) => {
    if (isOpen && !healthFetched.value) {
        void fetchHealthNight()
    }
})

async function fetchHealthNight(): Promise<void> {
    if (healthFetched.value) return
    healthFetched.value = true
    const dayDate = session.value?.therapy_day
    if (!dayDate) return
    healthLoading.value = true
    try {
        healthNight.value = await getHealthNight(dayDate)
    } catch (err: unknown) {
        if (isAxiosError(err) && err.response?.status === 404) {
            healthNotFound.value = true
        } else {
            healthError.value =
                err instanceof Error ? err.message : 'Failed to load Apple Health data'
        }
    } finally {
        healthLoading.value = false
    }
}

watch(
    sessionIdRef,
    async (newId) => {
        const gen = ++loadGeneration
        session.value = null
        events.value = []
        error.value = null
        loading.value = true
        selectedType.value = ''
        multiMode.value = false
        currentZoomRange.value = null
        healthOpen.value = false
        healthFetched.value = false
        healthLoading.value = false
        healthNotFound.value = false
        healthError.value = null
        healthNight.value = null
        resetWaveformData()
        cacheRegistry.clear()
        void loadDates()

        try {
            const s = await getSession(newId)
            if (gen !== loadGeneration) return
            session.value = s

            selectedType.value = s.waveform_types.includes('flow')
                ? 'flow'
                : (s.waveform_types[0] ?? '')
            // watcher triggers loadWindow() for the selected type

            if (s.has_event_data) {
                try {
                    const evts = await getSessionEvents(newId)
                    if (gen !== loadGeneration) return
                    events.value = evts
                } catch {
                    // Events failed — session still renders with empty events panel
                }
            }
        } catch (err: unknown) {
            if (gen !== loadGeneration) return
            error.value = err instanceof Error ? err.message : 'Failed to load session'
        } finally {
            if (gen === loadGeneration) loading.value = false
        }
    },
    { immediate: true },
)
</script>

<style scoped>
.session-detail {
    max-width: 1200px;
}

.session-header {
    margin-bottom: 1.5rem;
}

.session-title-nav {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin-bottom: 0.5rem;
}

.session-title-trigger {
    display: inline-flex;
    align-items: center;
    gap: 0.375rem;
    font-size: 1.5rem;
    font-weight: 600;
    line-height: 1.2;
    background: transparent;
    border: none;
    padding: 0;
    color: var(--color-foreground);
    cursor: pointer;
    transition: opacity 0.15s;
}

.session-title-trigger:hover {
    opacity: 0.75;
}

.nav-arrow {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    background: transparent;
    border: none;
    border-radius: 0.375rem;
    padding: 0.25rem;
    color: var(--color-foreground);
    cursor: pointer;
    transition: background-color 0.15s;
}

.nav-arrow:hover:not(:disabled) {
    background-color: var(--color-accent);
}

.nav-arrow:disabled {
    opacity: 0.3;
    cursor: default;
}

.session-meta {
    display: flex;
    align-items: center;
    gap: 1rem;
    flex-wrap: wrap;
    font-size: 0.9rem;
}

.stats-section {
    margin-bottom: 1.5rem;
}

.stats-section h2 {
    font-size: 1.1rem;
    font-weight: 600;
    margin-bottom: 0.75rem;
}

.stats-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
    gap: 0.75rem;
}

.stats-collapsible {
    margin-bottom: 0.5rem;
}

.collapsible-header {
    display: flex;
    width: 100%;
    align-items: center;
    justify-content: space-between;
    border-radius: 0.5rem;
    border: 1px solid var(--color-border);
    background: var(--color-card);
    padding: 0.75rem 1rem;
    text-align: left;
    font-weight: 600;
    font-size: 0.95rem;
    cursor: pointer;
}

.collapsible-header:hover {
    background: var(--color-accent);
}

.settings-panel {
    margin-bottom: 1.5rem;
}

.settings-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.4rem 2rem;
}

.quality-notes {
    margin-top: 0.75rem;
}

.quality-notes-label {
    font-size: 0.8rem;
    margin-bottom: 0.4rem;
}

.quality-chips {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
}

.quality-chip {
    display: inline-block;
    padding: 0.2rem 0.6rem;
    border-radius: 0.25rem;
    font-size: 0.8rem;
    color: var(--color-warning);
    border: 1px solid var(--color-warning);
    background: color-mix(in srgb, var(--color-warning) 10%, transparent);
}

.setting-row {
    display: flex;
    justify-content: space-between;
    font-size: 0.875rem;
    padding: 0.3rem 0;
    border-bottom: 1px solid var(--color-border);
}

.mask-info {
    display: flex;
    align-items: baseline;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-bottom: 1rem;
}

@media (max-width: 767.98px) {
    .session-title-nav {
        flex-wrap: wrap;
    }

    .session-title-trigger {
        font-size: 1.25rem;
    }

    .nav-arrow {
        padding: 0.75rem;
    }

    .settings-grid {
        grid-template-columns: 1fr;
    }

    .setting-row {
        overflow-wrap: anywhere;
    }
}
</style>
