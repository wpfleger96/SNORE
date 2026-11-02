<template>
    <div class="account-view">
        <h1 class="page-title">Account</h1>

        <p v-if="isDemo" role="alert" class="demo-banner">Demo accounts are read-only.</p>

        <div v-if="meLoading" class="loading-state">
            <Loader2 class="h-5 w-5 animate-spin" />
            <span>Loading…</span>
        </div>

        <div v-else-if="meError" class="error-state">
            <span>{{ meError }}</span>
        </div>

        <template v-else-if="me">
            <!-- Account card -->
            <div class="section-card">
                <h2>Account</h2>

                <div class="field-row">
                    <span class="field-label">Email</span>
                    <span class="field-value">{{ me.email }}</span>
                    <span class="role-badge">{{ me.role }}</span>
                </div>

                <form class="inline-form" @submit.prevent="saveDisplayName">
                    <label for="display-name" class="field-label">Display name</label>
                    <div class="inline-row">
                        <input
                            id="display-name"
                            v-model="displayName"
                            type="text"
                            class="field-input"
                            placeholder="Optional display name"
                            :disabled="isDemo || displayNameSaving"
                        />
                        <Button type="submit" :disabled="isDemo || displayNameSaving">
                            <Loader2 v-if="displayNameSaving" class="h-4 w-4 animate-spin" />
                            <span v-else>Save</span>
                        </Button>
                    </div>
                    <p v-if="displayNameError" role="alert" class="form-error">
                        {{ displayNameError }}
                    </p>
                    <p v-if="displayNameSuccess" class="form-success">{{ displayNameSuccess }}</p>
                </form>
            </div>

            <!-- Password card -->
            <div class="section-card">
                <h2>{{ me.has_password ? 'Password' : 'Set password' }}</h2>

                <form class="stacked-form" @submit.prevent="savePassword">
                    <div v-if="me.has_password" class="field-group">
                        <label for="current-password" class="field-label">Current password</label>
                        <input
                            id="current-password"
                            v-model="currentPassword"
                            type="password"
                            autocomplete="current-password"
                            class="field-input"
                            :disabled="isDemo || passwordSaving"
                        />
                    </div>

                    <div class="field-group">
                        <label for="new-password" class="field-label">New password</label>
                        <input
                            id="new-password"
                            v-model="newPassword"
                            type="password"
                            autocomplete="new-password"
                            class="field-input"
                            required
                            :disabled="isDemo || passwordSaving"
                        />
                    </div>

                    <div class="field-group">
                        <label for="confirm-password" class="field-label">Confirm password</label>
                        <input
                            id="confirm-password"
                            v-model="confirmPassword"
                            type="password"
                            autocomplete="new-password"
                            class="field-input"
                            required
                            :disabled="isDemo || passwordSaving"
                        />
                    </div>

                    <p v-if="passwordError" role="alert" class="form-error">{{ passwordError }}</p>
                    <p v-if="passwordSuccess" class="form-success">{{ passwordSuccess }}</p>

                    <Button type="submit" class="self-start" :disabled="isDemo || passwordSaving">
                        <Loader2 v-if="passwordSaving" class="h-4 w-4 animate-spin" />
                        <span v-else>{{
                            me.has_password ? 'Update password' : 'Set password'
                        }}</span>
                    </Button>
                </form>
            </div>

            <!-- Sign-in methods card -->
            <div class="section-card">
                <h2>Sign-in methods</h2>

                <div class="field-row">
                    <span class="field-label">Google</span>
                    <span class="field-value">{{
                        me.google_linked ? 'Linked' : 'Not linked'
                    }}</span>
                </div>

                <p v-if="connectSuccess" class="form-success">{{ connectSuccess }}</p>
                <p v-if="connectError" role="alert" class="form-error">{{ connectError }}</p>

                <template v-if="me.google_linked">
                    <p v-if="!me.has_password" class="form-error" role="alert">
                        Set a password first — Google is currently your only way to sign in.
                    </p>
                    <Button
                        v-else
                        variant="destructive"
                        class="self-start"
                        :disabled="isDemo || unlinkDialogOpen"
                        @click="unlinkDialogOpen = true"
                    >
                        Unlink Google
                    </Button>
                    <p v-if="unlinkError" role="alert" class="form-error">{{ unlinkError }}</p>
                </template>
                <template v-else>
                    <GoogleSignInButton
                        class="connect-google-btn"
                        :href="isDemo ? undefined : '/api/v1/auth/google/connect'"
                        :disabled="isDemo"
                        label="Connect Google"
                    />
                </template>

                <DeleteConfirmDialog
                    :visible="unlinkDialogOpen"
                    title="Unlink Google"
                    message="This will remove Google sign-in from your account and sign you out of all sessions. You will need to sign back in with your password."
                    :loading="false"
                    :deleting="unlinking"
                    @update:visible="unlinkDialogOpen = $event"
                    @confirm="confirmUnlinkGoogle"
                />
            </div>

            <!-- Two-factor authentication card — password accounts only -->
            <div v-if="me.has_password && !isDemo" class="section-card">
                <h2>Two-factor authentication</h2>

                <div v-if="totpLoading" class="loading-state">
                    <Loader2 class="h-4 w-4 animate-spin" />
                    <span>Loading…</span>
                </div>

                <div v-else-if="totpLoadError" class="error-state">
                    <span>{{ totpLoadError }}</span>
                </div>

                <!-- Not enrolled -->
                <template v-else-if="totpStatus && !totpStatus.enabled">
                    <template v-if="!showingEnrollWizard">
                        <p class="field-hint">
                            Adds a second step to password sign-in using an authenticator app.
                        </p>
                        <Button class="self-start" @click="startEnrollment">
                            Set up two-factor auth
                        </Button>
                        <p v-if="totpActionError" role="alert" class="form-error">
                            {{ totpActionError }}
                        </p>
                    </template>
                    <template v-else>
                        <TotpEnrollmentWizard @done="onEnrollDone" />
                        <button
                            type="button"
                            class="action-btn action-btn--ghost"
                            @click="showingEnrollWizard = false"
                        >
                            Cancel
                        </button>
                    </template>
                </template>

                <!-- Enrolled -->
                <template v-else-if="totpStatus && totpStatus.enabled">
                    <div class="field-row">
                        <span class="field-label">Status</span>
                        <span class="field-value totp-enabled-badge">Enabled</span>
                    </div>
                    <div class="field-row">
                        <span class="field-label">Recovery codes</span>
                        <span
                            class="field-value"
                            :class="{
                                'totp-low-codes': (totpStatus.recovery_codes_remaining ?? 0) <= 3,
                            }"
                        >
                            {{ totpStatus.recovery_codes_remaining ?? 0 }} remaining
                            <span
                                v-if="(totpStatus.recovery_codes_remaining ?? 0) <= 3"
                                class="low-codes-warning"
                            >
                                — Few recovery codes left, regenerate soon
                            </span>
                        </span>
                    </div>

                    <!-- Regenerate flow -->
                    <template v-if="showingRegenWizard">
                        <p class="field-hint">
                            Enter your current authenticator code to regenerate recovery codes:
                        </p>
                        <form class="stacked-form" @submit.prevent="submitRegenCodes">
                            <input
                                v-model="regenCode"
                                type="text"
                                inputmode="numeric"
                                pattern="[0-9]{6}"
                                maxlength="6"
                                placeholder="123456"
                                class="field-input totp-code-input"
                                autocomplete="one-time-code"
                                :disabled="regenBusy"
                            />
                            <p v-if="regenError" role="alert" class="form-error">
                                {{ regenError }}
                            </p>
                            <div class="inline-row">
                                <Button
                                    type="submit"
                                    :disabled="regenBusy || regenCode.length !== 6"
                                >
                                    <Loader2 v-if="regenBusy" class="h-4 w-4 animate-spin" />
                                    <span v-else>Regenerate</span>
                                </Button>
                                <button
                                    type="button"
                                    class="action-btn action-btn--ghost"
                                    :disabled="regenBusy"
                                    @click="cancelRegen"
                                >
                                    Cancel
                                </button>
                            </div>
                        </form>
                        <!-- Show new codes after regeneration -->
                        <template v-if="newRecoveryCodes.length > 0">
                            <p class="field-hint">
                                <strong>Save these new recovery codes.</strong> Your old codes are
                                now invalid.
                            </p>
                            <RecoveryCodesDisplay
                                :codes="newRecoveryCodes"
                                v-model="regenAcknowledged"
                            />
                            <Button
                                class="self-start"
                                :disabled="!regenAcknowledged"
                                @click="finishRegen"
                            >
                                Done
                            </Button>
                        </template>
                    </template>

                    <!-- Disable flow -->
                    <template v-else-if="showingDisableForm">
                        <p class="field-hint">
                            Enter your current password and authenticator code (or a recovery code)
                            to disable two-factor auth:
                        </p>
                        <form class="stacked-form" @submit.prevent="submitDisable">
                            <div class="field-group">
                                <label for="disable-password" class="field-label">Password</label>
                                <input
                                    id="disable-password"
                                    v-model="disablePassword"
                                    type="password"
                                    autocomplete="current-password"
                                    class="field-input"
                                    :disabled="disableBusy"
                                />
                            </div>
                            <div class="field-group">
                                <label for="disable-code" class="field-label">Code</label>
                                <input
                                    id="disable-code"
                                    v-model="disableCode"
                                    type="text"
                                    class="field-input totp-code-input"
                                    placeholder="123456 or recovery code"
                                    autocomplete="one-time-code"
                                    :disabled="disableBusy"
                                />
                            </div>
                            <p v-if="disableError" role="alert" class="form-error">
                                {{ disableError }}
                            </p>
                            <div class="inline-row">
                                <Button
                                    type="submit"
                                    variant="destructive"
                                    :disabled="disableBusy || !disablePassword || !disableCode"
                                >
                                    <Loader2 v-if="disableBusy" class="h-4 w-4 animate-spin" />
                                    <span v-else>Disable 2FA</span>
                                </Button>
                                <button
                                    type="button"
                                    class="action-btn action-btn--ghost"
                                    :disabled="disableBusy"
                                    @click="cancelDisable"
                                >
                                    Cancel
                                </button>
                            </div>
                        </form>
                    </template>

                    <!-- Enrolled idle state — action buttons -->
                    <template v-else>
                        <div class="totp-enrolled-actions">
                            <Button variant="outline" @click="showingRegenWizard = true">
                                Regenerate recovery codes
                            </Button>
                            <Button variant="destructive" @click="showingDisableForm = true">
                                Disable two-factor auth
                            </Button>
                        </div>
                        <p v-if="totpActionError" role="alert" class="form-error">
                            {{ totpActionError }}
                        </p>
                    </template>
                </template>
            </div>

            <!-- Preferences card -->
            <div class="section-card">
                <h2>Preferences</h2>

                <div v-if="prefsLoading" class="loading-state">
                    <Loader2 class="h-4 w-4 animate-spin" />
                    <span>Loading preferences…</span>
                </div>

                <div v-else-if="prefsError" class="error-state">
                    <span>{{ prefsError }}</span>
                </div>

                <form v-else class="stacked-form" @submit.prevent="savePreferences">
                    <div class="field-group">
                        <label for="landing-page" class="field-label">Landing page</label>
                        <select
                            id="landing-page"
                            v-model="prefLandingPage"
                            class="field-select"
                            :disabled="isDemo || prefSaving"
                        >
                            <option value="dashboard">Dashboard</option>
                            <option value="sessions">Sessions</option>
                            <option value="stats">Statistics</option>
                        </select>
                    </div>

                    <div class="field-group">
                        <label for="date-format" class="field-label">Date format</label>
                        <select
                            id="date-format"
                            v-model="prefDateFormat"
                            class="field-select"
                            :disabled="isDemo || prefSaving"
                        >
                            <option value="iso">ISO (2026-08-05)</option>
                            <option value="locale">Browser locale</option>
                            <option value="short">Short (8/5/26)</option>
                        </select>
                    </div>

                    <p v-if="prefSaveError" role="alert" class="form-error">{{ prefSaveError }}</p>
                    <p v-if="prefSuccess" class="form-success">{{ prefSuccess }}</p>

                    <Button type="submit" class="self-start" :disabled="isDemo || prefSaving">
                        <Loader2 v-if="prefSaving" class="h-4 w-4 animate-spin" />
                        <span v-else>Save preferences</span>
                    </Button>
                </form>
            </div>

            <!-- Timezone card -->
            <div class="section-card">
                <h2>Timezone</h2>
                <p class="field-hint">Applies to: {{ activeProfileName }}</p>

                <div v-if="tzLoading" class="loading-state">
                    <Loader2 class="h-4 w-4 animate-spin" />
                    <span>Loading…</span>
                </div>

                <form v-else class="stacked-form" @submit.prevent="saveTimezone">
                    <div class="field-group">
                        <label for="profile-timezone" class="field-label">Timezone</label>
                        <select
                            id="profile-timezone"
                            v-model="tzValue"
                            class="field-select"
                            :disabled="isDemo || tzSaving"
                        >
                            <option value="">Select timezone…</option>
                            <option v-for="tz in timezoneOptions" :key="tz" :value="tz">
                                {{ tz }}
                            </option>
                        </select>
                    </div>

                    <p v-if="!tzBaseline && detectedTimezone" class="tz-suggestion">
                        Detected: {{ detectedTimezone }}
                        <button
                            type="button"
                            class="action-btn action-btn--ghost"
                            @click="applyDetectedTimezone"
                        >
                            Use this
                        </button>
                    </p>

                    <p v-if="tzError" role="alert" class="form-error">{{ tzError }}</p>
                    <p v-if="tzSuccess" class="form-success">{{ tzSuccess }}</p>

                    <Button type="submit" class="self-start" :disabled="isDemo || tzSaving">
                        <Loader2 v-if="tzSaving" class="h-4 w-4 animate-spin" />
                        <span v-else>Save timezone</span>
                    </Button>
                </form>
            </div>

            <!-- Danger zone card — hidden for demo accounts -->
            <div v-if="!isDemo" class="section-card danger-zone-card">
                <h2>Danger Zone</h2>

                <div class="danger-zone-row">
                    <div>
                        <p class="danger-zone-title">Delete all my data</p>
                        <p class="danger-zone-subtitle">
                            Permanently removes all sleep sessions, waveforms, events, and analysis
                            results across all your profiles. Your account, profile containers, and
                            preferences are kept — you can re-import data afterward.
                        </p>
                    </div>
                    <Button
                        variant="destructive"
                        :disabled="deletingData"
                        @click="deleteDataDialogOpen = true"
                    >
                        <Loader2 v-if="deletingData" class="h-4 w-4 animate-spin" />
                        <span v-else>Delete all my data</span>
                    </Button>
                </div>

                <p v-if="deletingData" role="status" class="field-hint">
                    Deletion is running on the server and can take several minutes.
                </p>
                <p v-if="deleteDataSuccess" class="form-success">{{ deleteDataSuccess }}</p>
                <p v-if="deleteDataError" role="alert" class="form-error">{{ deleteDataError }}</p>
            </div>
        </template>
    </div>

    <!-- Delete-data confirmation dialog -->
    <DeleteConfirmDialog
        v-if="!isDemo"
        v-model:visible="deleteDataDialogOpen"
        title="Delete All My Data"
        message="This will permanently delete all sleep sessions, waveforms, events, and analysis across all your profiles. Your account and preferences are kept. This cannot be undone."
        :loading="false"
        :deleting="deletingData"
        confirm-phrase="delete"
        @confirm="handleDeleteData"
    />
</template>

<script setup lang="ts">
import { ref, computed, watch, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { Loader2 } from '@lucide/vue'
import { Button } from '@/components/ui/button'
import DeleteConfirmDialog from '@/components/DeleteConfirmDialog.vue'
import GoogleSignInButton from '@/components/GoogleSignInButton.vue'
import TotpEnrollmentWizard from '@/components/TotpEnrollmentWizard.vue'
import RecoveryCodesDisplay from '@/components/RecoveryCodesDisplay.vue'
import { useApiLoad } from '@/composables/useApiLoad'
import { useAuth } from '@/composables/useAuth'
import { useDateFormat } from '@/composables/useDateFormat'
import { useResetJob } from '@/composables/useResetJob'
import {
    getMe,
    updateDisplayName,
    changePassword,
    unlinkGoogle,
    getPreferences,
    updatePreferences,
    deleteMyData,
} from '@/api/me'
import { getTotpStatus, disableTotp, regenerateRecoveryCodes } from '@/api/totp'
import { listProfiles, setProfileTimezone } from '@/api/profiles'
import type { components } from '@/types/generated'

type TotpStatusResponse = components['schemas']['TotpStatusResponse']

type UserPreferences = components['schemas']['UserPreferences']

const router = useRouter()
const route = useRoute()
const { role, refreshStatus, clearAuth, activeProfileId, profiles } = useAuth()
const isDemo = computed(() => role.value === 'demo')
const { setDateFormat } = useDateFormat()

// ── Account ──────────────────────────────────────────────────────────────────

const { data: me, loading: meLoading, error: meError } = useApiLoad(getMe)

const displayName = ref('')
const displayNameSaving = ref(false)
const displayNameError = ref<string | null>(null)
const displayNameSuccess = ref<string | null>(null)

watch(me, (val) => {
    if (val) displayName.value = val.display_name ?? ''
})

async function saveDisplayName() {
    if (isDemo.value) return
    displayNameError.value = null
    displayNameSuccess.value = null
    displayNameSaving.value = true
    try {
        await updateDisplayName({ display_name: displayName.value.trim() || null })
        await refreshStatus()
        displayNameSuccess.value = 'Display name updated'
    } catch (e: unknown) {
        displayNameError.value = e instanceof Error ? e.message : 'Failed to update display name'
    } finally {
        displayNameSaving.value = false
    }
}

// ── Password ──────────────────────────────────────────────────────────────────

const currentPassword = ref('')
const newPassword = ref('')
const confirmPassword = ref('')
const passwordSaving = ref(false)
const passwordError = ref<string | null>(null)
const passwordSuccess = ref<string | null>(null)

async function savePassword() {
    if (isDemo.value) return
    passwordError.value = null
    passwordSuccess.value = null
    if (!newPassword.value) {
        passwordError.value = 'New password cannot be empty'
        return
    }
    if (newPassword.value !== confirmPassword.value) {
        passwordError.value = 'Passwords do not match'
        return
    }
    passwordSaving.value = true
    let succeeded = false
    try {
        await changePassword({
            new_password: newPassword.value,
            ...(me.value?.has_password ? { current_password: currentPassword.value } : {}),
        })
        succeeded = true
        passwordSuccess.value = 'Password updated'
        currentPassword.value = ''
        newPassword.value = ''
        confirmPassword.value = ''
    } catch (e: unknown) {
        passwordError.value = e instanceof Error ? e.message : 'Failed to change password'
    } finally {
        passwordSaving.value = false
    }
    // A successful change means a password now exists — update locally instead
    // of re-fetching, so a network hiccup can't blank the page or the success
    // banner (useApiLoad surfaces reload failures via meError, which swaps the
    // whole view into its error state).
    if (succeeded && me.value) {
        me.value = { ...me.value, has_password: true }
    }
}

// ── Sign-in methods (Google unlink) ──────────────────────────────────────────

const unlinkDialogOpen = ref(false)
const unlinking = ref(false)
const unlinkError = ref<string | null>(null)
const connectError = ref<string | null>(null)
const connectSuccess = ref<string | null>(null)

async function confirmUnlinkGoogle() {
    if (isDemo.value) return
    unlinkError.value = null
    unlinking.value = true
    try {
        await unlinkGoogle()
        clearAuth()
        router.push('/')
    } catch (e: unknown) {
        unlinkError.value = e instanceof Error ? e.message : 'Failed to unlink Google'
        unlinkDialogOpen.value = false
    } finally {
        unlinking.value = false
    }
}

// ── Two-factor authentication ─────────────────────────────────────────────────

const totpStatus = ref<TotpStatusResponse | null>(null)
const totpLoading = ref(false)
const totpLoadError = ref<string | null>(null)
const totpActionError = ref<string | null>(null)

const showingEnrollWizard = ref(false)

// Regenerate flow
const showingRegenWizard = ref(false)
const regenCode = ref('')
const regenBusy = ref(false)
const regenError = ref<string | null>(null)
const newRecoveryCodes = ref<string[]>([])
const regenAcknowledged = ref(false)

// Disable flow
const showingDisableForm = ref(false)
const disablePassword = ref('')
const disableCode = ref('')
const disableBusy = ref(false)
const disableError = ref<string | null>(null)

async function loadTotpStatus() {
    if (isDemo.value) return
    totpLoading.value = true
    totpLoadError.value = null
    try {
        totpStatus.value = await getTotpStatus()
    } catch (e: unknown) {
        totpLoadError.value = e instanceof Error ? e.message : 'Failed to load 2FA status'
    } finally {
        totpLoading.value = false
    }
}

function startEnrollment() {
    totpActionError.value = null
    showingEnrollWizard.value = true
}

async function onEnrollDone() {
    showingEnrollWizard.value = false
    await loadTotpStatus()
}

async function submitRegenCodes() {
    regenError.value = null
    regenBusy.value = true
    try {
        const result = await regenerateRecoveryCodes({ code: regenCode.value })
        newRecoveryCodes.value = result.recovery_codes
        regenCode.value = ''
    } catch (e: unknown) {
        regenError.value = e instanceof Error ? e.message : 'Failed to regenerate codes'
    } finally {
        regenBusy.value = false
    }
}

function cancelRegen() {
    showingRegenWizard.value = false
    regenCode.value = ''
    regenError.value = null
    newRecoveryCodes.value = []
    regenAcknowledged.value = false
}

function finishRegen() {
    cancelRegen()
    void loadTotpStatus()
}

async function submitDisable() {
    disableError.value = null
    disableBusy.value = true
    try {
        await disableTotp({
            password: disablePassword.value,
            code: disableCode.value.trim().toLowerCase(),
        })
        // Backend clears the session cookie — sign the user out locally and redirect.
        clearAuth()
        router.push('/')
    } catch (e: unknown) {
        disableError.value = e instanceof Error ? e.message : 'Failed to disable 2FA'
    } finally {
        disableBusy.value = false
    }
}

function cancelDisable() {
    showingDisableForm.value = false
    disablePassword.value = ''
    disableCode.value = ''
    disableError.value = null
}

// ── Preferences ───────────────────────────────────────────────────────────────

const { data: prefs, loading: prefsLoading, error: prefsError } = useApiLoad(getPreferences)

const prefLandingPage = ref<UserPreferences['landing_page']>('dashboard')
const prefDateFormat = ref<UserPreferences['date_format']>('iso')
const prefBaseline = ref<UserPreferences | null>(null)
const prefSaving = ref(false)
const prefSaveError = ref<string | null>(null)
const prefSuccess = ref<string | null>(null)

watch(prefs, (val) => {
    if (val) {
        prefLandingPage.value = val.landing_page
        prefDateFormat.value = val.date_format
        prefBaseline.value = { landing_page: val.landing_page, date_format: val.date_format }
    }
})

async function savePreferences() {
    if (isDemo.value) return
    prefSaveError.value = null
    prefSuccess.value = null

    const update: components['schemas']['UserPreferencesUpdate'] = {}
    if (prefBaseline.value?.landing_page !== prefLandingPage.value) {
        update.landing_page = prefLandingPage.value
    }
    if (prefBaseline.value?.date_format !== prefDateFormat.value) {
        update.date_format = prefDateFormat.value
    }
    if (Object.keys(update).length === 0) {
        prefSuccess.value = 'No changes to save'
        return
    }

    prefSaving.value = true
    try {
        const result = await updatePreferences(update)
        prefBaseline.value = { landing_page: result.landing_page, date_format: result.date_format }
        prefLandingPage.value = result.landing_page
        prefDateFormat.value = result.date_format
        if (update.date_format !== undefined) {
            setDateFormat(result.date_format)
        }
        prefSuccess.value = 'Preferences saved'
    } catch (e: unknown) {
        prefSaveError.value = e instanceof Error ? e.message : 'Failed to save preferences'
    } finally {
        prefSaving.value = false
    }
}

// ── Profile Timezone ──────────────────────────────────────────────────────────

const activeProfileName = computed(
    () => profiles.value.find((p) => p.id === activeProfileId.value)?.name ?? 'active profile',
)

const timezoneOptions = computed<string[]>(() => {
    try {
        const intl = Intl as typeof Intl & { supportedValuesOf?: (key: string) => string[] }
        return intl.supportedValuesOf?.('timeZone') ?? []
    } catch {
        return []
    }
})

const detectedTimezone = computed<string | null>(() => {
    try {
        return Intl.DateTimeFormat().resolvedOptions().timeZone ?? null
    } catch {
        return null
    }
})

const tzValue = ref('')
const tzBaseline = ref<string | null>(null)
const tzLoading = ref(true)
const tzSaving = ref(false)
const tzError = ref<string | null>(null)
const tzSuccess = ref<string | null>(null)

async function loadProfileTimezone() {
    if (!activeProfileId.value) return
    tzLoading.value = true
    tzError.value = null
    try {
        const allProfiles = await listProfiles()
        const active = allProfiles.find((p) => p.id === activeProfileId.value)
        const tz = active?.timezone ?? null
        tzValue.value = tz ?? ''
        tzBaseline.value = tz
    } catch (e: unknown) {
        tzError.value = e instanceof Error ? e.message : 'Failed to load timezone'
    } finally {
        tzLoading.value = false
    }
}

watch(activeProfileId, () => {
    loadProfileTimezone()
})

async function saveTimezone() {
    if (isDemo.value || !activeProfileId.value) return
    tzError.value = null
    tzSuccess.value = null

    const newTz = tzValue.value.trim() || null
    if (newTz === tzBaseline.value) {
        tzSuccess.value = 'No changes to save'
        return
    }

    tzSaving.value = true
    try {
        const result = await setProfileTimezone(activeProfileId.value, newTz)
        const tz = result.timezone ?? null
        tzValue.value = tz ?? ''
        tzBaseline.value = tz
        tzSuccess.value = newTz ? 'Timezone saved' : 'Timezone cleared'
    } catch (e: unknown) {
        tzError.value = e instanceof Error ? e.message : 'Failed to save timezone'
    } finally {
        tzSaving.value = false
    }
}

function applyDetectedTimezone() {
    if (detectedTimezone.value) {
        tzValue.value = detectedTimezone.value
    }
}

onMounted(() => {
    loadProfileTimezone()
    void loadTotpStatus()
    if ('google_connected' in route.query) {
        connectSuccess.value = 'Google account linked'
        void router.replace({ path: '/account' })
    } else if ('google_connect_error' in route.query) {
        connectError.value =
            "Couldn't link Google. Make sure the Google account's email matches your SNORE account email."
        void router.replace({ path: '/account' })
    }
})

// ── Delete all my data ────────────────────────────────────────────────────────

const deleteDataDialogOpen = ref(false)
const {
    running: deletingData,
    status: deleteDataStatus,
    error: deleteDataError,
    start: startDeleteDataJob,
} = useResetJob('delete_data')

const deleteDataSuccess = computed(() => {
    const result = deleteDataStatus.value?.delete_data_result
    if (!result) return null
    return `Deleted: ${result.devices_deleted} device(s), ${result.import_jobs_deleted} import record(s) across ${result.profiles_processed} profile(s).`
})

async function handleDeleteData(): Promise<void> {
    if (isDemo.value) return
    deleteDataDialogOpen.value = false
    await startDeleteDataJob(deleteMyData)
}
</script>

<style scoped>
.account-view {
    max-width: 560px;
    margin: 0 auto;
    padding: 1.5rem;
}

.demo-banner {
    background: hsl(from var(--color-warning) h s l / 0.12);
    border: 1px solid hsl(from var(--color-warning) h s l / 0.4);
    border-radius: 6px;
    color: var(--color-warning);
    font-size: 0.875rem;
    font-weight: 500;
    padding: 0.625rem 0.875rem;
    margin-bottom: 1.25rem;
}

.field-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
}

/* min-width aligns the Email label in its flex row */
.field-label {
    min-width: 7rem;
}

.field-value {
    font-size: 0.875rem;
    color: var(--color-muted-foreground);
}

.role-badge {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.15rem 0.5rem;
    border-radius: 999px;
    background: hsl(from var(--color-primary) h s l / 0.1);
    color: var(--color-primary);
}

.inline-form {
    display: flex;
    flex-direction: column;
    gap: 0.5rem;
}

.inline-row {
    display: flex;
    gap: 0.5rem;
}

.stacked-form {
    display: flex;
    flex-direction: column;
    gap: 0.75rem;
}

/* flex: 1 so the input fills the inline-row */
.field-input {
    flex: 1;
}

.form-error {
    font-size: 0.875rem;
    color: var(--color-destructive);
    margin: 0;
}

.form-success {
    font-size: 0.875rem;
    color: var(--color-success);
    margin: 0;
}

.danger-zone-card {
    border-color: var(--color-destructive);
}

.danger-zone-row {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 1rem;
    flex-wrap: wrap;
}

.danger-zone-title {
    font-size: 0.875rem;
    font-weight: 600;
    color: var(--color-foreground);
    margin: 0 0 0.25rem;
}

.danger-zone-subtitle {
    font-size: 0.8rem;
    color: var(--color-muted-foreground);
    margin: 0;
    max-width: 36ch;
}

.field-hint {
    font-size: 0.8rem;
    color: var(--color-muted-foreground);
    margin: 0 0 0.75rem;
}

.tz-suggestion {
    font-size: 0.8rem;
    color: var(--color-muted-foreground);
    display: flex;
    align-items: center;
    gap: 0.5rem;
    margin: 0;
}

.action-btn {
    background: none;
    border: 1px solid var(--color-border);
    border-radius: 4px;
    padding: 0.125rem 0.5rem;
    font-size: 0.8rem;
    cursor: pointer;
    color: var(--color-foreground);
}

.action-btn--ghost {
    border-color: transparent;
    color: var(--color-primary);
}

.action-btn--ghost:hover {
    background: hsl(from var(--color-primary) h s l / 0.08);
}

.totp-enabled-badge {
    color: var(--color-success, var(--color-primary));
    font-weight: 500;
}

.totp-low-codes {
    color: var(--color-warning, var(--color-destructive));
}

.low-codes-warning {
    font-size: 0.8rem;
}

.totp-code-input {
    font-family: monospace;
    letter-spacing: 0.1em;
    width: 12rem;
}

.totp-enrolled-actions {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
}

.field-group {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

.connect-google-btn {
    width: auto;
    align-self: flex-start;
}

@media (max-width: 767.98px) {
    /* Stack label above value instead of a 7rem label column */
    .field-row {
        flex-direction: column;
        align-items: flex-start;
        gap: 0.25rem;
    }

    .field-label {
        min-width: 0;
    }

    /* Comfortable tap target — mirrors the shared layout.css mobile rule,
       which this scoped .action-btn otherwise out-specifies */
    .action-btn {
        padding: 0.5rem 0.875rem;
    }

    .totp-code-input {
        width: 100%;
        max-width: 12rem;
    }
}
</style>
