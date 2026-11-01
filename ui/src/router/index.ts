import { createRouter, createWebHistory } from 'vue-router'
import type { RouteLocationNormalized } from 'vue-router'
import { useAuth } from '@/composables/useAuth'
import { getPreferences } from '@/api/me'
import type { components } from '@/types/generated'

type UserPreferences = components['schemas']['UserPreferences']

declare module 'vue-router' {
    interface RouteMeta {
        authFree?: boolean
        requiresMultiuser?: boolean
        requiresAdmin?: boolean
    }
}

// landing_page value → route path.
const LANDING_PAGE_MAP: Record<UserPreferences['landing_page'], string> = {
    dashboard: '/dashboard',
    sessions: '/sessions',
    stats: '/stats',
}

/** Resolve the authenticated user's preferred landing path (multiuser only). */
export async function resolveLandingPath(): Promise<string> {
    const prefs = await getPreferences(AbortSignal.timeout(5000)).catch(() => null)
    return LANDING_PAGE_MAP[prefs?.landing_page ?? 'dashboard'] ?? '/dashboard'
}

function sessionIdProp(route: { params: { id: string | string[] } }) {
    const id = Number(route.params.id)
    return { sessionId: Number.isFinite(id) ? id : -1 }
}

function dayDateProp(route: { params: { date: string | string[] } }) {
    const { date } = route.params
    return { dayDate: Array.isArray(date) ? date[0] : date }
}

const router = createRouter({
    history: createWebHistory(),
    routes: [
        // Auth-free routes — no sidebar, no navigation chrome.
        {
            path: '/',
            name: 'login',
            component: () => import('@/views/LoginView.vue'),
            meta: { authFree: true },
        },
        {
            path: '/invite',
            name: 'invite',
            component: () => import('@/views/InviteView.vue'),
            meta: { authFree: true },
        },

        // Authenticated routes
        {
            path: '/dashboard',
            name: 'dashboard',
            component: () => import('@/views/DashboardView.vue'),
        },
        {
            path: '/profiles',
            name: 'profiles',
            component: () => import('@/views/ProfilesView.vue'),
        },
        {
            path: '/sessions',
            name: 'sessions',
            component: () => import('@/views/SessionListView.vue'),
        },
        {
            path: '/equipment',
            name: 'equipment',
            component: () => import('@/views/EquipmentView.vue'),
        },
        { path: '/devices', redirect: '/equipment' },
        {
            path: '/sessions/:id',
            name: 'session-detail',
            component: () => import('@/views/SessionDetailView.vue'),
            props: sessionIdProp,
        },
        {
            path: '/sessions/:id/events',
            name: 'session-events',
            component: () => import('@/views/EventExplorerView.vue'),
            props: sessionIdProp,
        },
        {
            path: '/sessions/:id/analysis',
            name: 'session-analysis',
            component: () => import('@/views/AnalysisView.vue'),
            props: sessionIdProp,
        },
        {
            path: '/stats',
            name: 'stats',
            component: () => import('@/views/StatsView.vue'),
        },
        {
            path: '/rx',
            name: 'rx-history',
            component: () => import('@/views/RxHistoryView.vue'),
        },
        {
            path: '/import',
            name: 'import',
            component: () => import('@/views/ImportView.vue'),
        },
        {
            path: '/export',
            name: 'export',
            component: () => import('@/views/ExportView.vue'),
        },
        {
            path: '/reports',
            name: 'reports',
            component: () => import('@/views/ReportsView.vue'),
        },
        {
            path: '/analysis',
            name: 'analysis-management',
            component: () => import('@/views/AnalysisManagementView.vue'),
        },
        {
            path: '/database',
            name: 'database',
            component: () => import('@/views/DatabaseView.vue'),
            meta: { requiresAdmin: true },
        },
        {
            path: '/about',
            name: 'about',
            component: () => import('@/views/AboutView.vue'),
        },
        {
            path: '/validation',
            name: 'validation',
            component: () => import('@/views/ValidationView.vue'),
        },
        {
            path: '/days/:date',
            name: 'day-detail',
            component: () => import('@/views/DayDetailView.vue'),
            props: dayDateProp,
        },
        {
            path: '/days/:date/experimental',
            name: 'day-experimental',
            component: () => import('@/views/DayExperimentalView.vue'),
            props: dayDateProp,
        },
        {
            path: '/apple-health',
            name: 'apple-health',
            component: () => import('@/views/AppleHealthView.vue'),
        },
        {
            path: '/apple-health/:date',
            name: 'apple-health-night',
            component: () => import('@/views/AppleHealthNightView.vue'),
            props: (route: RouteLocationNormalized) => ({ nightDate: route.params.date }),
        },

        // Multiuser-only routes
        {
            path: '/account',
            name: 'account',
            component: () => import('@/views/AccountView.vue'),
            meta: { requiresMultiuser: true },
        },
        {
            path: '/totp-enroll',
            name: 'totp-enroll',
            component: () => import('@/views/TotpEnrollView.vue'),
            meta: { requiresMultiuser: true },
        },
        {
            path: '/admin/users',
            name: 'admin-users',
            component: () => import('@/views/AdminUsersView.vue'),
            meta: { requiresMultiuser: true, requiresAdmin: true },
        },
        {
            path: '/admin/mcp',
            name: 'admin-mcp',
            component: () => import('@/views/AdminMcpView.vue'),
            meta: { requiresMultiuser: true, requiresAdmin: true },
        },
    ],
})

// Exported for unit testing — the production router uses this directly.
export async function authGuard(to: RouteLocationNormalized): Promise<string | boolean | void> {
    const { fetchStatus, isAuthenticated, isLocal, role, totpEnrollmentRequired } = useAuth()

    try {
        await fetchStatus()
    } catch {
        // fetchStatus is non-rejecting by contract; this catch is defensive-only
        // and cannot be triggered by the real composable.
    }

    const authed = isAuthenticated.value || isLocal.value

    // Unauthenticated user on a guarded route → login.
    if (!authed && !to.meta.authFree) {
        return '/'
    }

    // Forced TOTP enrollment: the server rejects everything except the enrollment
    // endpoints, so short-circuit all destinations before landing-path resolution.
    if (authed && !isLocal.value && totpEnrollmentRequired.value) {
        return to.path === '/totp-enroll' ? undefined : '/totp-enroll'
    }

    // Authenticated user landing on login page → preference-based landing or dashboard.
    if (authed && to.path === '/') {
        if (isLocal.value) {
            // Local mode has no preferences endpoint; skip the fetch.
            return '/dashboard'
        }
        return resolveLandingPath()
    }

    // Multiuser-only routes are inaccessible in local mode.
    if (to.meta.requiresMultiuser && isLocal.value) {
        return '/dashboard'
    }

    // Admin-only routes require the admin role.
    if (to.meta.requiresAdmin && role.value !== 'admin') {
        return '/dashboard'
    }

    // Prevent stale visits to /totp-enroll when enrollment is no longer required
    // (the enrollment-required case already short-circuited above).
    if (authed && !isLocal.value && to.path === '/totp-enroll') {
        return resolveLandingPath()
    }
}

router.beforeEach(authGuard)

export default router
