<template>
    <aside class="app-sidebar">
        <div class="sidebar-header">
            <RouterLink to="/dashboard" class="sidebar-logo">SNORE</RouterLink>
        </div>
        <nav class="sidebar-nav">
            <span class="nav-group-label">Data</span>
            <RouterLink to="/dashboard" class="nav-item">
                <BarChart3 class="h-4 w-4" />
                <span>Dashboard</span>
            </RouterLink>
            <RouterLink to="/sessions" class="nav-item">
                <List class="h-4 w-4" />
                <span>Sessions</span>
            </RouterLink>
            <RouterLink to="/equipment" class="nav-item">
                <HardDrive class="h-4 w-4" />
                <span>Equipment</span>
            </RouterLink>
            <RouterLink to="/apple-health" class="nav-item">
                <Moon class="h-4 w-4" />
                <span>Apple Health</span>
            </RouterLink>

            <span class="nav-group-label">Analysis</span>
            <RouterLink to="/analysis" class="nav-item">
                <Brain class="h-4 w-4" />
                <span>Analysis</span>
            </RouterLink>
            <RouterLink to="/validation" class="nav-item">
                <CheckCircle class="h-4 w-4" />
                <span>Validation</span>
            </RouterLink>

            <span class="nav-group-label">Tools</span>
            <RouterLink v-if="canWrite" to="/import" class="nav-item">
                <Upload class="h-4 w-4" />
                <span>Import</span>
            </RouterLink>
            <RouterLink to="/export" class="nav-item">
                <Download class="h-4 w-4" />
                <span>Export</span>
            </RouterLink>
            <RouterLink to="/reports" class="nav-item">
                <FileText class="h-4 w-4" />
                <span>Reports</span>
            </RouterLink>
            <span class="nav-group-label">Settings</span>
            <RouterLink to="/stats" class="nav-item">
                <TrendingUp class="h-4 w-4" />
                <span>Stats</span>
            </RouterLink>
            <RouterLink to="/rx" class="nav-item">
                <Pill class="h-4 w-4" />
                <span>RX History</span>
            </RouterLink>

            <template v-if="role === 'admin'">
                <span class="nav-group-label">Admin</span>
                <RouterLink to="/database" class="nav-item">
                    <Database class="h-4 w-4" />
                    <span>Database</span>
                </RouterLink>
                <RouterLink v-if="!isLocal" to="/admin/users" class="nav-item">
                    <Users class="h-4 w-4" />
                    <span>Users</span>
                </RouterLink>
                <RouterLink v-if="!isLocal" to="/admin/mcp" class="nav-item">
                    <Server class="h-4 w-4" />
                    <span>MCP Server</span>
                </RouterLink>
            </template>
        </nav>
        <div class="sidebar-footer">
            <!-- User menu — multiuser mode only -->
            <template v-if="!isLocal && isAuthenticated">
                <DropdownMenu>
                    <DropdownMenuTrigger as-child>
                        <button
                            class="nav-item user-trigger"
                            type="button"
                            @click="fetchGoogleStatus"
                        >
                            <User class="h-4 w-4 shrink-0" />
                            <span class="user-name">{{ displayName }}</span>
                            <ChevronUp class="h-3 w-3 shrink-0 ml-auto" />
                        </button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent side="top" align="start" class="w-52">
                        <DropdownMenuLabel class="text-xs text-muted-foreground">
                            Profiles
                        </DropdownMenuLabel>
                        <DropdownMenuItem
                            v-for="profile in profiles"
                            :key="profile.id"
                            class="profile-menu-item"
                            :class="{ 'profile-menu-item--active': profile.id === activeProfileId }"
                            @click="switchToProfile(profile.id)"
                        >
                            <Check
                                v-if="profile.id === activeProfileId"
                                class="h-3.5 w-3.5 mr-1.5 shrink-0"
                            />
                            <span v-else class="h-3.5 w-3.5 mr-1.5 shrink-0 inline-block" />
                            {{ profile.name }}
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem as-child>
                            <RouterLink to="/profiles" class="dropdown-link">
                                <Settings class="h-4 w-4 mr-2" />
                                Manage profiles
                            </RouterLink>
                        </DropdownMenuItem>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem as-child>
                            <RouterLink to="/account" class="dropdown-link">
                                <UserCog class="h-4 w-4 mr-2" />
                                Account
                            </RouterLink>
                        </DropdownMenuItem>
                        <DropdownMenuLabel v-if="googleLinked !== null" class="google-status-label">
                            Google: {{ googleLinked ? 'Linked' : 'Not linked' }}
                        </DropdownMenuLabel>
                        <DropdownMenuSeparator />
                        <DropdownMenuItem @click="handleLogout" class="text-destructive">
                            <LogOut class="h-4 w-4 mr-2" />
                            Sign out
                        </DropdownMenuItem>
                    </DropdownMenuContent>
                </DropdownMenu>
            </template>
            <!-- Status unknown (status is null, so isLocal is necessarily false): show muted placeholder -->
            <div v-else-if="statusUnknown" class="reconnecting">Reconnecting…</div>

            <RouterLink to="/about" class="nav-item">
                <Info class="h-4 w-4" />
                <span>About</span>
            </RouterLink>
            <button class="nav-item" @click="toggleDark">
                <Sun v-if="isDark" class="h-4 w-4" />
                <Moon v-else class="h-4 w-4" />
                <span>{{ isDark ? 'Light Mode' : 'Dark Mode' }}</span>
            </button>
        </div>
    </aside>
</template>

<script setup lang="ts">
import { computed, ref, watch } from 'vue'
import { useRouter } from 'vue-router'
import {
    BarChart3,
    Brain,
    Check,
    CheckCircle,
    ChevronUp,
    Database,
    Download,
    FileText,
    HardDrive,
    Info,
    List,
    LogOut,
    Moon,
    Pill,
    Server,
    Settings,
    Sun,
    TrendingUp,
    Upload,
    User,
    UserCog,
    Users,
} from '@lucide/vue'
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuItem,
    DropdownMenuLabel,
    DropdownMenuSeparator,
    DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { useDarkMode } from '@/composables/useDarkMode'
import { useAuth } from '@/composables/useAuth'
import { getMe } from '@/api/me'

const router = useRouter()
const { isDark, toggleDark } = useDarkMode()
const {
    user,
    isAuthenticated,
    isLocal,
    profiles,
    activeProfileId,
    canWrite,
    role,
    statusUnknown,
    setActiveProfile,
    logout,
} = useAuth()

const displayName = computed(() => user.value?.display_name || user.value?.email || 'Account')

// Fetched lazily on first dropdown open — avoids an extra request on every page load.
const googleLinked = ref<boolean | null>(null)
let _googleFetched = false
let _googleFetchAttempts = 0

async function fetchGoogleStatus(): Promise<void> {
    if (_googleFetched || _googleFetchAttempts >= 2) return
    _googleFetchAttempts++
    _googleFetched = true
    try {
        const me = await getMe()
        googleLinked.value = me.google_linked
    } catch {
        // Silently ignore — the dropdown still opens without the Google status line.
        // Retries are capped at 2 total attempts to avoid a storm on persistent errors.
        _googleFetched = false
    }
}

// Reset stale cached status on logout so the next session sees fresh data.
watch(isAuthenticated, (v) => {
    if (!v) {
        _googleFetched = false
        _googleFetchAttempts = 0
        googleLinked.value = null
    }
})

async function switchToProfile(profileId: number) {
    if (profileId === activeProfileId.value) return
    await setActiveProfile(profileId)
}

async function handleLogout() {
    await logout()
    router.push('/')
}
</script>

<style scoped>
.app-sidebar {
    background: var(--color-card);
    border-right: 1px solid var(--color-border);
    display: flex;
    flex-direction: column;
    position: sticky;
    top: 0;
    height: 100dvh;
    overflow-y: auto;
}

/* Inside the mobile Sheet: larger nav items for ~44px tap targets. */
@media (max-width: 767.98px) {
    .nav-item {
        padding: 0.75rem;
    }
}

.sidebar-header {
    padding: 1.25rem 1rem;
    border-bottom: 1px solid var(--color-border);
}

.sidebar-logo {
    font-size: 1.1rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    color: var(--color-primary);
    text-decoration: none;
}

.sidebar-nav {
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
    padding: 0.75rem 0.5rem;
}

.nav-item {
    display: flex;
    align-items: center;
    gap: 0.625rem;
    padding: 0.6rem 0.75rem;
    border-radius: 6px;
    font-size: 0.9rem;
    text-decoration: none;
    color: var(--color-foreground);
    transition: background 0.15s;
    cursor: pointer;
    background: none;
    border: none;
    width: 100%;
    text-align: left;
}

.nav-item:hover {
    background: var(--color-accent);
}

.router-link-active {
    background: hsl(from var(--color-primary) h s l / 0.1);
    color: var(--color-primary);
    font-weight: 500;
}

.sidebar-footer {
    margin-top: auto;
    padding: 0.75rem 0.5rem;
    border-top: 1px solid var(--color-border);
    display: flex;
    flex-direction: column;
    gap: 0.25rem;
}

.nav-group-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--color-muted-foreground);
    padding: 0.75rem 0.75rem 0.25rem;
}

.nav-group-label:first-child {
    padding-top: 0;
}

.user-trigger {
    overflow: hidden;
}

.user-name {
    flex: 1;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    font-size: 0.875rem;
}

.profile-menu-item {
    cursor: pointer;
}

.profile-menu-item--active {
    font-weight: 500;
}

.dropdown-link {
    display: flex;
    align-items: center;
    width: 100%;
    text-decoration: none;
    color: inherit;
}

.reconnecting {
    padding: 0.6rem 0.75rem;
    font-size: 0.875rem;
    color: var(--color-muted-foreground);
}

.google-status-label {
    padding-left: 2.25rem;
    font-size: 0.75rem;
    font-weight: normal;
    color: var(--color-muted-foreground);
}
</style>
