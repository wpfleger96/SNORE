import { computed, ref } from 'vue'
import type { AuthStatusResponse } from '@/types'
import {
    demoLoginUser,
    getAuthStatus,
    loginUser,
    logoutUser,
    submitTotpChallenge,
    switchProfile,
} from '@/api/auth'
import { clearStoredResetJobs } from '@/composables/useResetJob'

// Module-level singleton — shared across all callers (same pattern as useDarkMode).
const status = ref<AuthStatusResponse | null>(null)
// In-memory only; a page refresh abandons any pending challenge (user logs in again).
const totpPendingToken = ref<string | null>(null)
const profileKey = ref(0)
let _fetchPromise: Promise<void> | null = null
let _lastFetched = 0
let _generation = 0
let _healTimer: ReturnType<typeof setTimeout> | null = null
let _healDelay = 3_000 // doubles on each consecutive failed heal, capped at 30s; reset on success/clearAuth/refreshStatus

const REVALIDATE_MS = 5 * 60 * 1000
// Fast first path — keeps the login form snappy when the origin is healthy.
const AUTH_STATUS_TIMEOUT_MS = 3_000
// Longer retry — survives SQLite write-lock windows / container restarts up to ~11.5s total.
const AUTH_STATUS_RETRY_TIMEOUT_MS = 8_000

export function useAuth() {
    const user = computed(() => status.value?.user ?? null)
    const isAuthenticated = computed(() => status.value?.authenticated ?? false)
    const isLocal = computed(() => status.value?.auth_mode === 'local')
    const profiles = computed(() => status.value?.profiles ?? [])
    const activeProfileId = computed(() => status.value?.active_profile_id ?? null)
    const authMode = computed(() => status.value?.auth_mode ?? null)
    const role = computed(() => status.value?.user?.role ?? null)
    // canWrite: trust the server-reported role — an authenticated non-demo actor
    // can write in any auth mode.
    const canWrite = computed(() => role.value !== 'demo' && isAuthenticated.value)
    const demoAvailable = computed(() => status.value?.demo_available ?? false)
    const totpEnrollmentRequired = computed(() => status.value?.totp_enrollment_required ?? false)
    // Distinct from !isAuthenticated: true while auth state is unknown (fetch pending or failed).
    const statusUnknown = computed(() => status.value === null)

    async function fetchStatus(): Promise<void> {
        const now = Date.now()
        if (status.value !== null && now - _lastFetched < REVALIDATE_MS) return
        if (_fetchPromise !== null) return _fetchPromise
        const gen = _generation

        async function attempt(gen: number): Promise<AuthStatusResponse> {
            try {
                return await getAuthStatus(AbortSignal.timeout(AUTH_STATUS_TIMEOUT_MS))
            } catch {
                // Retry once after a short backoff.
                await new Promise<void>((r) => setTimeout(r, 500))
                if (_generation !== gen) throw new Error('superseded')
                return getAuthStatus(AbortSignal.timeout(AUTH_STATUS_RETRY_TIMEOUT_MS))
            }
        }

        _fetchPromise = attempt(gen)
            .then((s) => {
                if (_generation === gen) {
                    // Only write if this generation is still the active one.
                    status.value = s
                    _lastFetched = Date.now()
                    _healDelay = 3_000 // reset backoff on success
                } else if (_fetchPromise !== null) {
                    // Superseded by a newer generation — chain to the active fetch
                    // so callers awaiting this promise get the authenticated result.
                    return _fetchPromise
                }
            })
            .catch(() => {
                // Both network attempts failed — schedule background recovery if still unknown.
                // When superseded (generation mismatch), the guard below prevents scheduling;
                // the superseding fetch owns recovery, and callers self-correct via reactivity.
                if (_generation === gen && status.value === null) {
                    if (_healTimer !== null) clearTimeout(_healTimer)
                    const delay = _healDelay
                    _healDelay = Math.min(_healDelay * 2, 30_000)
                    _healTimer = setTimeout(() => {
                        _healTimer = null
                        if (_generation === gen && status.value === null) {
                            void fetchStatus()
                        }
                    }, delay)
                }
            })
            .finally(() => {
                // Only clear _fetchPromise if this generation owns it —
                // prevents a stale .finally() from killing a newer in-flight request.
                if (_generation === gen) {
                    _fetchPromise = null
                }
            })
        return _fetchPromise
    }

    async function refreshStatus(): Promise<void> {
        if (_healTimer !== null) {
            clearTimeout(_healTimer)
            _healTimer = null
        }
        status.value = null
        _fetchPromise = null
        _generation++ // invalidate any in-flight requests from earlier generation
        _lastFetched = 0
        _healDelay = 3_000 // reset backoff when caller explicitly refreshes
        await fetchStatus()
    }

    async function login(email: string, password: string): Promise<{ totpRequired: boolean }> {
        const result = await loginUser({ email, password })
        if (result.totp_required) {
            if (!result.pending_token) {
                throw new Error('Server error: missing pending token')
            }
            totpPendingToken.value = result.pending_token
            return { totpRequired: true }
        }
        totpPendingToken.value = null
        await refreshStatus()
        return { totpRequired: false }
    }

    async function submitTotp(code: string): Promise<void> {
        const token = totpPendingToken.value
        if (!token) throw new Error('No pending TOTP challenge')
        await submitTotpChallenge({ pending_token: token, code: code.trim().toLowerCase() })
        totpPendingToken.value = null
        await refreshStatus()
    }

    function clearTotpChallenge(): void {
        totpPendingToken.value = null
    }

    async function demoLogin(): Promise<void> {
        await demoLoginUser()
        await refreshStatus()
    }

    function clearAuth(): void {
        if (_healTimer !== null) {
            clearTimeout(_healTimer)
            _healTimer = null
        }
        _healDelay = 3_000 // reset backoff on explicit clear
        status.value = null
        totpPendingToken.value = null
        _fetchPromise = null
        _lastFetched = 0
        _generation++
        clearStoredResetJobs()
    }

    async function logout(): Promise<void> {
        try {
            await logoutUser()
        } finally {
            clearAuth()
        }
    }

    async function setActiveProfile(profileId: number): Promise<void> {
        await switchProfile({ profile_id: profileId })
        if (status.value) {
            status.value = { ...status.value, active_profile_id: profileId }
        }
        profileKey.value++
    }

    return {
        user,
        isAuthenticated,
        isLocal,
        profiles,
        activeProfileId,
        authMode,
        role,
        canWrite,
        demoAvailable,
        totpEnrollmentRequired,
        statusUnknown,
        profileKey,
        fetchStatus,
        refreshStatus,
        login,
        submitTotp,
        clearTotpChallenge,
        demoLogin,
        logout,
        clearAuth,
        setActiveProfile,
    }
}
