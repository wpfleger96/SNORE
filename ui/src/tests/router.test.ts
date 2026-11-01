import { beforeEach, describe, expect, it, vi } from 'vitest'
import { mount } from '@vue/test-utils'
import { createRouter, createWebHistory } from 'vue-router'
import { ref } from 'vue'

vi.mock('@/composables/useAuth')
vi.mock('@/api/me', () => ({ getPreferences: vi.fn() }))

// Import the production guard — guard logic comes from the real module, not a copy.
// The default export (router instance) is used in the route resolution tests below.
import router, { authGuard } from '@/router'
import { useAuth } from '@/composables/useAuth'
import { getPreferences } from '@/api/me'
import { makeAuthMock as baseMakeAuthMock } from './helpers/mockUseAuth'

function makeAuthMock(authed: boolean, local: boolean) {
    vi.mocked(useAuth).mockReturnValue(
        baseMakeAuthMock({
            isAuthenticated: ref(authed) as never,
            isLocal: ref(local) as never,
            authMode: ref(local ? 'local' : 'multiuser') as never,
        }) as never,
    )
}

function makeRoute(
    path: string,
    meta: Record<string, unknown> = {},
): { path: string; meta: Record<string, unknown> } {
    return { path, meta }
}

describe('authGuard (production)', () => {
    beforeEach(() => {
        vi.resetAllMocks()
        // resolveLandingPath() calls getPreferences; mock it so redirect destinations are deterministic.
        vi.mocked(getPreferences).mockResolvedValue({ landing_page: 'dashboard' } as never)
    })

    it('test_unauthenticated_access_to_guarded_route_redirects_to_login', async () => {
        makeAuthMock(false, false)
        const result = await authGuard(makeRoute('/sessions') as never)
        expect(result).toBe('/')
    })

    it('test_unauthenticated_access_to_auth_free_route_is_allowed', async () => {
        makeAuthMock(false, false)
        const result = await authGuard(makeRoute('/', { authFree: true }) as never)
        expect(result).toBeUndefined()
    })

    it('test_unauthenticated_access_to_invite_route_is_allowed', async () => {
        makeAuthMock(false, false)
        const result = await authGuard(makeRoute('/invite', { authFree: true }) as never)
        expect(result).toBeUndefined()
    })

    it('test_authenticated_access_to_login_redirects_to_dashboard', async () => {
        makeAuthMock(true, false)
        const result = await authGuard(makeRoute('/') as never)
        expect(result).toBe('/dashboard')
    })

    it('test_local_mode_bypasses_auth_check', async () => {
        makeAuthMock(false, true)
        const result = await authGuard(makeRoute('/sessions') as never)
        expect(result).toBeUndefined()
    })

    it('test_local_mode_on_login_page_redirects_to_dashboard', async () => {
        makeAuthMock(false, true)
        const result = await authGuard(makeRoute('/') as never)
        expect(result).toBe('/dashboard')
    })

    it('test_database_multiuser_member_redirects_to_dashboard', async () => {
        vi.mocked(useAuth).mockReturnValue(
            baseMakeAuthMock({
                isAuthenticated: ref(true) as never,
                isLocal: ref(false) as never,
                role: ref('member') as never,
            }) as never,
        )
        const result = await authGuard(makeRoute('/database', { requiresAdmin: true }) as never)
        expect(result).toBe('/dashboard')
    })

    it('test_database_multiuser_admin_is_allowed', async () => {
        vi.mocked(useAuth).mockReturnValue(
            baseMakeAuthMock({
                isAuthenticated: ref(true) as never,
                isLocal: ref(false) as never,
                role: ref('admin') as never,
            }) as never,
        )
        const result = await authGuard(makeRoute('/database', { requiresAdmin: true }) as never)
        expect(result).toBeUndefined()
    })

    it('test_database_local_mode_is_allowed', async () => {
        vi.mocked(useAuth).mockReturnValue(
            baseMakeAuthMock({
                isAuthenticated: ref(false) as never,
                isLocal: ref(true) as never,
                role: ref('admin') as never,
            }) as never,
        )
        const result = await authGuard(makeRoute('/database', { requiresAdmin: true }) as never)
        expect(result).toBeUndefined()
    })

    it('test_authGuard_defensive_catch_allows_through_when_fetchStatus_rejects', async () => {
        // fetchStatus is non-rejecting by contract; this exercises the defensive catch
        // path in authGuard that cannot be triggered by the real composable.
        vi.mocked(useAuth).mockReturnValue(
            baseMakeAuthMock({
                fetchStatus: vi.fn().mockRejectedValueOnce(new Error('Network')),
                isAuthenticated: ref(false) as never,
                isLocal: ref(false) as never,
            }) as never,
        )
        await expect(
            authGuard(makeRoute('/invite', { authFree: true }) as never),
        ).resolves.toBeUndefined()
    })

    it('test_enrollment_required_redirects_to_totp_enroll', async () => {
        vi.mocked(useAuth).mockReturnValue(
            baseMakeAuthMock({
                isAuthenticated: ref(true) as never,
                isLocal: ref(false) as never,
                totpEnrollmentRequired: ref(true) as never,
            }) as never,
        )
        const result = await authGuard(makeRoute('/sessions') as never)
        expect(result).toBe('/totp-enroll')
    })

    it('test_enrollment_required_does_not_redirect_to_totp_enroll_when_already_there', async () => {
        vi.mocked(useAuth).mockReturnValue(
            baseMakeAuthMock({
                isAuthenticated: ref(true) as never,
                isLocal: ref(false) as never,
                totpEnrollmentRequired: ref(true) as never,
            }) as never,
        )
        const result = await authGuard(makeRoute('/totp-enroll') as never)
        expect(result).toBeUndefined()
    })

    it('test_totp_enroll_redirects_away_when_enrollment_not_required', async () => {
        vi.mocked(useAuth).mockReturnValue(
            baseMakeAuthMock({
                isAuthenticated: ref(true) as never,
                isLocal: ref(false) as never,
                totpEnrollmentRequired: ref(false) as never,
            }) as never,
        )
        vi.mocked(getPreferences).mockResolvedValueOnce({ landing_page: 'dashboard' } as never)
        const result = await authGuard(makeRoute('/totp-enroll') as never)
        // Guard calls resolveLandingPath() → getPreferences() → '/dashboard'.
        expect(result).toBe('/dashboard')
    })

    it('test_enrollment_required_does_not_apply_in_local_mode', async () => {
        vi.mocked(useAuth).mockReturnValue(
            baseMakeAuthMock({
                isAuthenticated: ref(false) as never,
                isLocal: ref(true) as never,
                totpEnrollmentRequired: ref(true) as never,
            }) as never,
        )
        const result = await authGuard(makeRoute('/sessions') as never)
        // Local mode users bypass TOTP enforcement — no redirect to /totp-enroll
        expect(result).toBeUndefined()
    })
})

// ---------------------------------------------------------------------------
// Workflow integration tests
// ---------------------------------------------------------------------------

vi.mock('@/api/auth')
vi.mock('@/api/profiles')
vi.mock('@/api/import', () => ({
    importFiles: vi.fn(),
    importHealthFile: vi.fn(),
    precheckFiles: vi.fn().mockResolvedValue(new Set()),
    isImportableFile: vi.fn().mockReturnValue(true),
    isAnchorFile: vi.fn().mockReturnValue(false),
    detectSources: vi.fn(),
    importFromPath: vi.fn(),
}))
vi.mock('@/api/importJobs', () => ({
    getImportJobs: vi.fn().mockResolvedValue({ jobs: [] }),
    cancelImport: vi.fn(),
}))
vi.mock('@/composables/useDarkMode', () => ({
    useDarkMode: () => ({ isDark: ref(false), toggleDark: vi.fn() }),
}))
vi.mock('@/utils/formatting', () => ({ formatBytes: (n: number) => `${n}B` }))
vi.mock('@/api/health', () => ({
    getHealthNights: vi.fn().mockResolvedValue({ items: [], total: 0, limit: 20, offset: 0 }),
    getHealthNight: vi.fn(),
    getHealthNightSamples: vi.fn(),
    getHealthNightDates: vi.fn().mockResolvedValue({ dates: [] }),
}))

import * as authApi from '@/api/auth'
import * as profilesApi from '@/api/profiles'
import * as importApi from '@/api/import'

describe('workflow: invite route renders from fragment', () => {
    it('test_invite_route_renders_from_hash', async () => {
        vi.mocked(authApi.lookupInvite).mockResolvedValueOnce({
            valid: true,
            email: 'invited@example.com',
        })
        makeAuthMock(false, false)

        const InviteView = (await import('@/views/InviteView.vue')).default
        const testRouter = createRouter({
            history: createWebHistory(),
            routes: [{ path: '/invite', component: InviteView }],
        })
        await testRouter.push('/invite#SOMETOKEN')

        mount(InviteView, { global: { plugins: [testRouter] } })

        // Await onMounted which calls lookupInvite with the fragment token.
        await new Promise((r) => setTimeout(r, 0))

        // lookupInvite is called with the token extracted from route.hash (#SOMETOKEN → "SOMETOKEN").
        expect(authApi.lookupInvite).toHaveBeenCalledWith({ token: 'SOMETOKEN' })
    })
})

describe('workflow: redemption refreshes status', () => {
    it('test_redemption_clears_cache_and_reaches_dashboard', async () => {
        // refreshStatus is the key: it clears stale cached unauthenticated status
        const refreshStatusMock = vi.fn().mockResolvedValue(undefined)
        vi.mocked(useAuth).mockReturnValue(
            baseMakeAuthMock({ refreshStatus: refreshStatusMock }) as never,
        )

        vi.mocked(authApi.lookupInvite).mockResolvedValueOnce({ valid: true, email: 'u@x.com' })
        vi.mocked(authApi.redeemInvite).mockResolvedValueOnce({ message: 'ok' })

        const InviteView = (await import('@/views/InviteView.vue')).default
        const testRouter = createRouter({
            history: createWebHistory(),
            routes: [
                { path: '/invite', component: InviteView },
                { path: '/dashboard', component: { template: '<div />' } },
            ],
        })
        await testRouter.push('/invite#MYTOKEN')
        const wrapper = mount(InviteView, { global: { plugins: [testRouter] } })

        // Await onMounted lookup
        await wrapper.vm.$nextTick()
        await wrapper.vm.$nextTick()

        // Fill form and submit
        await wrapper.find('input[type="password"]').setValue('password1')
        await wrapper.findAll('input[type="password"]')[1].setValue('password1')
        await wrapper.find('form').trigger('submit')
        await wrapper.vm.$nextTick()
        await wrapper.vm.$nextTick()

        expect(refreshStatusMock).toHaveBeenCalled()
    })
})

describe('workflow: login page has no sidebar', () => {
    it('test_login_page_has_no_sidebar', async () => {
        makeAuthMock(false, false)
        vi.mocked(authApi.getAuthStatus).mockResolvedValue({
            authenticated: false,
            auth_mode: 'multiuser',
            profiles: [],
        } as never)

        const App = (await import('@/App.vue')).default
        const LoginView = (await import('@/views/LoginView.vue')).default
        const AppSidebar = (await import('@/components/AppSidebar.vue')).default

        const testRouter = createRouter({
            history: createWebHistory(),
            routes: [
                { path: '/', component: LoginView, meta: { authFree: true } },
                { path: '/dashboard', component: { template: '<div />' } },
            ],
        })
        await testRouter.push('/')

        const wrapper = mount(App, { global: { plugins: [testRouter] } })
        // Sidebar should not be rendered on auth-free routes
        expect(wrapper.findComponent(AppSidebar).exists()).toBe(false)
    })

    it('test_dashboard_has_app_layout_wrapper', async () => {
        makeAuthMock(true, false)
        vi.mocked(authApi.getAuthStatus).mockResolvedValue({
            authenticated: true,
            auth_mode: 'multiuser',
            profiles: [],
        } as never)

        const App = (await import('@/App.vue')).default
        const Dashboard = { template: '<div class="dash" />' }

        const testRouter = createRouter({
            history: createWebHistory(),
            routes: [{ path: '/dashboard', component: Dashboard }],
        })
        await testRouter.push('/dashboard')

        const wrapper = mount(App, { global: { plugins: [testRouter] } })
        // Protected route must render inside .app-layout so the CSS grid applies.
        expect(wrapper.find('.app-layout').exists()).toBe(true)
        expect(wrapper.find('.app-main').exists()).toBe(true)
    })
})

// profileKey increment is covered in useAuth.test.ts (setActiveProfile_incrementsProfileKey).
// Testing it here would require vi.unmock which breaks subsequent mocks in the same file.

describe('workflow: import passes selectedProfileId to importFiles', () => {
    it('test_import_passes_selected_profile_id_to_importFiles', async () => {
        // selectedProfileId (2) differs from the active session profile (1).
        // importFiles must be called with profileId=2 so the backend targets the right profile.
        // Falsifiability: if profileId is omitted from the importFiles call this assertion fails.
        vi.mocked(useAuth).mockReturnValue(
            baseMakeAuthMock({
                isAuthenticated: ref(true) as never,
                user: ref({ id: 1, email: 'u@x.com', display_name: 'U', role: 'user' }) as never,
                profiles: ref([
                    { id: 1, name: 'Primary' },
                    { id: 2, name: 'Work' },
                ]) as never,
                activeProfileId: ref(1) as never,
            }) as never,
        )

        vi.mocked(importApi.importFiles).mockResolvedValueOnce({ job_id: 'job-1' })

        const ImportView = (await import('@/views/ImportView.vue')).default
        const testRouter = createRouter({
            history: createWebHistory(),
            routes: [{ path: '/import', component: ImportView }],
        })
        await testRouter.push('/import')
        const wrapper = mount(ImportView, { global: { plugins: [testRouter] } })

        const vm = wrapper.vm as unknown as Record<string, unknown>
        // Select profile 2 (different from active profile 1)
        vm['selectedProfileId'] = 2
        await (vm as { handleImport: () => Promise<void> }).handleImport()

        // importFiles must receive the selected profile id as the third argument
        expect(importApi.importFiles).toHaveBeenCalledWith(
            expect.anything(), // fileEntries
            expect.anything(), // onProgress callback
            2, // selectedProfileId
        )
    })
})

describe('workflow: import does not call setActiveProfile during submit', () => {
    it('test_import_does_not_call_setActiveProfile', async () => {
        // After the pass-2 fix, setActiveProfile is no longer called from handleImport.
        // The profileKey remount that would unmount the view during upload is eliminated.
        const setActiveProfileMock = vi.fn()
        vi.mocked(useAuth).mockReturnValue(
            baseMakeAuthMock({
                isAuthenticated: ref(true) as never,
                user: ref({ id: 1, email: 'u@x.com', display_name: 'U', role: 'user' }) as never,
                profiles: ref([
                    { id: 1, name: 'Primary' },
                    { id: 2, name: 'Work' },
                ]) as never,
                activeProfileId: ref(1) as never,
                setActiveProfile: setActiveProfileMock,
            }) as never,
        )

        vi.mocked(importApi.importFiles).mockResolvedValueOnce({ job_id: 'job-1' })

        const ImportView = (await import('@/views/ImportView.vue')).default
        const testRouter = createRouter({
            history: createWebHistory(),
            routes: [{ path: '/import', component: ImportView }],
        })
        await testRouter.push('/import')
        const wrapper = mount(ImportView, { global: { plugins: [testRouter] } })

        // Select a different profile and call handleImport.
        const vm = wrapper.vm as unknown as Record<string, unknown>
        vm['selectedProfileId'] = 2
        await (vm as { handleImport: () => Promise<void> }).handleImport()

        // setActiveProfile must NOT be called — it would increment profileKey and
        // unmount the view before the upload starts.
        expect(setActiveProfileMock).not.toHaveBeenCalled()
    })
})

describe('workflow: profile default calls PATCH endpoint', () => {
    it('test_profile_default_calls_patch_endpoint', async () => {
        const setDefaultMock = vi.fn().mockResolvedValueOnce({
            id: 2,
            name: 'Work',
            user_id: 1,
            is_default: true,
            created_at: '2026-01-01T00:00:00Z',
        })
        vi.mocked(profilesApi.setDefaultProfile).mockImplementation(setDefaultMock)
        vi.mocked(profilesApi.listProfiles).mockResolvedValueOnce([
            {
                id: 1,
                name: 'Primary',
                user_id: 1,
                created_at: '2026-01-01T00:00:00Z',
                is_default: false,
            },
            {
                id: 2,
                name: 'Work',
                user_id: 1,
                created_at: '2026-01-01T00:00:00Z',
                is_default: false,
            },
        ])

        makeAuthMock(true, false)

        const ProfilesView = (await import('@/views/ProfilesView.vue')).default
        const testRouter = createRouter({
            history: createWebHistory(),
            routes: [{ path: '/profiles', component: ProfilesView }],
        })
        await testRouter.push('/profiles')
        const wrapper = mount(ProfilesView, { global: { plugins: [testRouter] } })

        // Wait for onMounted listProfiles
        await wrapper.vm.$nextTick()
        await wrapper.vm.$nextTick()

        // Click "Set default" button (should call setDefaultProfile, NOT setActiveProfile)
        const setDefaultBtn = wrapper
            .findAll('button')
            .find((b) => b.text().includes('Set default'))
        expect(setDefaultBtn).toBeDefined()
        await setDefaultBtn!.trigger('click')
        await wrapper.vm.$nextTick()

        expect(setDefaultMock).toHaveBeenCalled()
        // Must NOT have called setActiveProfile on the auth composable
        const authMock = vi.mocked(useAuth)()
        expect(authMock.setActiveProfile).not.toHaveBeenCalled()
    })
})

describe('workflow: profile CRUD refreshes shared auth store', () => {
    it('test_create_profile_calls_refreshStatus_to_update_shared_store', async () => {
        const refreshStatusMock = vi.fn().mockResolvedValue(undefined)
        vi.mocked(useAuth).mockReturnValue(
            baseMakeAuthMock({
                isAuthenticated: ref(true) as never,
                user: ref({ id: 1, email: 'u@x.com', display_name: 'U', role: 'user' }) as never,
                profiles: ref([{ id: 1, name: 'Primary' }]) as never,
                activeProfileId: ref(1) as never,
                refreshStatus: refreshStatusMock,
            }) as never,
        )

        vi.mocked(profilesApi.listProfiles).mockResolvedValueOnce([
            {
                id: 1,
                name: 'Primary',
                user_id: 1,
                created_at: '2026-01-01T00:00:00Z',
                is_default: false,
            },
        ])
        vi.mocked(profilesApi.createProfile).mockResolvedValueOnce({
            id: 2,
            name: 'Work',
            user_id: 1,
            created_at: '2026-01-01T00:00:00Z',
            is_default: false,
        })

        const ProfilesView = (await import('@/views/ProfilesView.vue')).default
        const testRouter = createRouter({
            history: createWebHistory(),
            routes: [{ path: '/profiles', component: ProfilesView }],
        })
        await testRouter.push('/profiles')
        const wrapper = mount(ProfilesView, { global: { plugins: [testRouter] } })

        // Wait for onMounted
        await wrapper.vm.$nextTick()
        await wrapper.vm.$nextTick()

        // Fill the create form and submit
        await wrapper.find('.create-form input').setValue('Work')
        await wrapper.find('.create-form').trigger('submit')
        await wrapper.vm.$nextTick()
        await wrapper.vm.$nextTick()

        // refreshStatus must be called so AppSidebar and ImportView pick up the new profile.
        expect(refreshStatusMock).toHaveBeenCalled()
    })
})

describe('workflow: profile mutation success not shadowed by refresh failure', () => {
    it('test_profile_rename_success_shows_no_error_when_refresh_throws', async () => {
        // A committed rename must not be reported as a failure just because the
        // subsequent GET /auth/status throws. refreshStatus() is fire-and-forget.
        const refreshStatusMock = vi.fn().mockRejectedValue(new Error('Network'))
        vi.mocked(useAuth).mockReturnValue(
            baseMakeAuthMock({
                isAuthenticated: ref(true) as never,
                user: ref({ id: 1, email: 'u@x.com', display_name: 'U', role: 'user' }) as never,
                profiles: ref([{ id: 1, name: 'Primary' }]) as never,
                activeProfileId: ref(1) as never,
                refreshStatus: refreshStatusMock,
            }) as never,
        )

        vi.mocked(profilesApi.listProfiles).mockResolvedValueOnce([
            {
                id: 1,
                name: 'Primary',
                user_id: 1,
                created_at: '2026-01-01T00:00:00Z',
                is_default: false,
            },
        ])
        vi.mocked(profilesApi.updateProfile).mockResolvedValueOnce({
            id: 1,
            name: 'NewName',
            user_id: 1,
            created_at: '2026-01-01T00:00:00Z',
            is_default: false,
        })

        const ProfilesView = (await import('@/views/ProfilesView.vue')).default
        const testRouter = createRouter({
            history: createWebHistory(),
            routes: [{ path: '/profiles', component: ProfilesView }],
        })
        await testRouter.push('/profiles')
        const wrapper = mount(ProfilesView, { global: { plugins: [testRouter] } })

        // Wait for onMounted
        await wrapper.vm.$nextTick()
        await wrapper.vm.$nextTick()

        // Click the Rename button for the first profile
        const renameBtn = wrapper.findAll('button').find((b) => b.text().includes('Rename'))
        expect(renameBtn).toBeDefined()
        await renameBtn!.trigger('click')
        await wrapper.vm.$nextTick()

        // Enter new name and save
        await wrapper.find('.profile-name-input').setValue('NewName')
        const saveBtn = wrapper.findAll('button').find((b) => b.text().includes('Save'))
        expect(saveBtn).toBeDefined()
        await saveBtn!.trigger('click')

        // Flush async: mutation resolves, refresh fires and forgets
        await wrapper.vm.$nextTick()
        await wrapper.vm.$nextTick()
        await wrapper.vm.$nextTick()

        // refreshStatus was called
        expect(refreshStatusMock).toHaveBeenCalled()
        // No error banner — refresh failure must not surface as a mutation failure
        expect(wrapper.find('.action-error').exists()).toBe(false)
        // Profile name updated in the local list from the mutation response
        expect(wrapper.text()).toContain('NewName')
    })
})

// ---------------------------------------------------------------------------
// Apple Health route registration
// ---------------------------------------------------------------------------

describe('routes: apple-health routes', () => {
    // Uses router.resolve() — no navigation, no component mount, no guards fire.
    it('test_apple_health_base_route_resolves_to_named_route', () => {
        const resolved = router.resolve('/apple-health')
        expect(resolved.name).toBe('apple-health')
    })

    it('test_apple_health_night_route_resolves_with_date_param', () => {
        const resolved = router.resolve('/apple-health/2024-01-15')
        expect(resolved.name).toBe('apple-health-night')
        expect(resolved.params.date).toBe('2024-01-15')
    })
})

describe('routes: day routes', () => {
    it('test_day_experimental_route_passes_date_to_view', () => {
        const resolved = router.resolve('/days/2026-05-03/experimental')
        expect(resolved.name).toBe('day-experimental')
        const props = resolved.matched[0].props.default as (r: typeof resolved) => unknown
        expect(props(resolved)).toEqual({ dayDate: '2026-05-03' })
    })
})
