<template>
    <section class="archive-card">
        <div class="archive-description">
            <p class="archive-title">Re-import from server archive</p>
            <p class="archive-subtitle">
                Import CPAP sessions again from the raw files already stored on the server, without
                uploading anything. Sessions that are already imported are skipped unless you force
                a re-parse.
            </p>
        </div>

        <div class="archive-options">
            <label class="archive-check">
                <input
                    v-model="force"
                    type="checkbox"
                    class="archive-checkbox"
                    data-testid="force"
                />
                <span>Force re-parse</span>
            </label>
            <p v-if="force" class="archive-warning" role="alert">
                <AlertTriangle class="h-4 w-4 shrink-0" />
                <span>{{ FORCE_WARNING }}</span>
            </p>

            <div class="archive-dates">
                <div class="archive-date">
                    <span class="archive-date-label">From</span>
                    <DatePickerInput v-model="fromDate" placeholder="Earliest" />
                </div>
                <div class="archive-date">
                    <span class="archive-date-label">To</span>
                    <DatePickerInput v-model="toDate" placeholder="Latest" />
                </div>
                <Button
                    v-if="fromDate || toDate"
                    variant="ghost"
                    size="sm"
                    class="archive-clear"
                    @click="clearDates"
                >
                    Clear dates
                </Button>
            </div>
            <p class="archive-hint">{{ DATE_HINT }}</p>
            <p v-if="dateOrderInvalid" class="error-text">
                The From date must be on or before the To date.
            </p>

            <label class="archive-check">
                <input
                    v-model="analyze"
                    type="checkbox"
                    class="archive-checkbox"
                    data-testid="analyze"
                />
                <span>Run analysis afterwards</span>
            </label>
        </div>

        <p v-if="error" class="error-text">{{ error }}</p>

        <div class="archive-actions">
            <Button
                :variant="force ? 'destructive' : 'default'"
                :disabled="pending || dateOrderInvalid"
                @click="onRescanClick"
            >
                Re-import from archive
            </Button>
        </div>

        <AlertDialog :open="confirmOpen" @update:open="confirmOpen = $event">
            <AlertDialogContent>
                <AlertDialogHeader>
                    <AlertDialogTitle>Force re-parse?</AlertDialogTitle>
                    <AlertDialogDescription>
                        Every already-imported session {{ scopeText }} will be deleted, including
                        its events, statistics, waveforms, settings, analysis results and breath
                        rows, and rebuilt from the files in the server archive. Disabled sessions
                        stay disabled. {{ analysisText }}
                        <template v-if="fromDate || toDate">{{ DATE_HINT }}</template>
                    </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                    <Button variant="destructive" :disabled="pending" @click="submit">
                        Delete and re-parse
                    </Button>
                </AlertDialogFooter>
            </AlertDialogContent>
        </AlertDialog>
    </section>
</template>

<script setup lang="ts">
import { computed, ref } from 'vue'
import { AlertTriangle } from '@lucide/vue'
import { Button } from '@/components/ui/button'
import {
    AlertDialog,
    AlertDialogCancel,
    AlertDialogContent,
    AlertDialogDescription,
    AlertDialogFooter,
    AlertDialogHeader,
    AlertDialogTitle,
} from '@/components/ui/alert-dialog'
import DatePickerInput from '@/components/DatePickerInput.vue'
import { triggerRescan } from '@/api/import'
import { formatDateFull } from '@/utils/formatting'

const props = defineProps<{ profileId: number | null }>()
const emit = defineEmits<{ started: [] }>()

const DATE_HINT =
    "Dates select nights, the date each night is filed under: a session that starts before noon belongs to the previous day's night."
const FORCE_WARNING =
    'Every matched session that is already imported is deleted and recreated, including its analysis results and breath rows. Disabled sessions stay disabled.'

const force = ref(false)
const fromDate = ref('')
const toDate = ref('')
const analyze = ref(true)
const pending = ref(false)
const error = ref<string | null>(null)
const confirmOpen = ref(false)

// ISO dates compare correctly as strings.
const dateOrderInvalid = computed(
    () => fromDate.value !== '' && toDate.value !== '' && fromDate.value > toDate.value,
)

const scopeText = computed(() => {
    const from = fromDate.value ? formatDateFull(fromDate.value) : null
    const to = toDate.value ? formatDateFull(toDate.value) : null
    if (from && to) return `from ${from} to ${to}`
    if (from) return `from ${from} onward`
    if (to) return `up to ${to}`
    return 'in the archive'
})

const analysisText = computed(() =>
    analyze.value
        ? 'Analysis runs again for the re-imported sessions.'
        : 'Analysis will not run; start it from the Analysis page when you need it.',
)

function clearDates(): void {
    fromDate.value = ''
    toDate.value = ''
}

function onRescanClick(): void {
    if (force.value) {
        confirmOpen.value = true
        return
    }
    void submit()
}

async function submit(): Promise<void> {
    if (pending.value) return
    pending.value = true
    error.value = null
    try {
        await triggerRescan({
            profile_id: props.profileId,
            force: force.value,
            from_date: fromDate.value || null,
            to_date: toDate.value || null,
            analyze: analyze.value,
        })
        confirmOpen.value = false
        // A forced re-parse is destructive; never leave it armed for the next run.
        force.value = false
        emit('started')
    } catch (e: unknown) {
        confirmOpen.value = false
        error.value = e instanceof Error ? e.message : 'Re-import from archive failed'
    } finally {
        pending.value = false
    }
}
</script>

<style scoped>
.archive-card {
    border: 1px solid var(--color-border);
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    background: var(--color-card);
    display: flex;
    flex-direction: column;
    gap: 0.85rem;
    margin-bottom: 1.5rem;
}

.archive-description {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
}

.archive-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: var(--color-foreground);
    margin: 0;
}

.archive-subtitle {
    font-size: 0.8rem;
    color: var(--color-muted-foreground);
    margin: 0;
}

.archive-options {
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
}

.archive-check {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.85rem;
    color: var(--color-foreground);
    cursor: pointer;
    align-self: flex-start;
}

.archive-checkbox {
    cursor: pointer;
}

.archive-warning {
    display: flex;
    align-items: flex-start;
    gap: 0.4rem;
    font-size: 0.8rem;
    color: var(--color-destructive);
    margin: 0;
}

.archive-dates {
    display: flex;
    align-items: flex-end;
    flex-wrap: wrap;
    gap: 0.75rem;
}

.archive-date {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

.archive-date-label {
    font-size: 0.75rem;
    color: var(--color-muted-foreground);
}

.archive-hint {
    font-size: 0.75rem;
    color: var(--color-muted-foreground);
    margin: 0;
}

.archive-actions {
    display: flex;
    justify-content: flex-end;
}

.error-text {
    font-size: 0.875rem;
    color: var(--color-destructive);
    margin: 0;
}

@media (max-width: 767.98px) {
    .archive-dates {
        flex-direction: column;
        align-items: stretch;
    }

    .archive-check {
        min-height: var(--tap-target);
    }

    .archive-actions {
        flex-direction: column;
        align-items: stretch;
    }

    .archive-actions > *,
    .archive-clear {
        min-height: var(--tap-target);
    }
}
</style>
