import { expect, test, type Page, type Route, type TestInfo } from '@playwright/test'
import {
    summaryFixture,
    experimentalStatsFixture,
    dayExperimentalFixture,
    trendsFixture,
    daysFixture,
    sessionsFixture,
    devicesFixture,
    deviceDetailFixture,
    periodsFixture,
    recordsFixture,
    rxHistoryFixture,
    rxCurrentFixture,
    rxCompareFixture,
    rxChangesFixture,
    rxAllFixture,
    maskLogFixture,
    maskEpochsFixture,
    sessionDetailFixture,
    sessionEventsFixture,
    flowWaveformFixture,
    dateListFixture,
    dataRangeFixture,
} from './tests/fixtures/api-fixtures'

function routeApi(route: Route) {
    const url = route.request().url()

    // Local-mode auth status so the router guard lets guarded routes render.
    if (url.includes('/auth/status'))
        return route.fulfill({
            json: {
                authenticated: false,
                auth_mode: 'local',
                user: null,
                profiles: [],
                active_profile_id: null,
                demo_available: false,
            },
        })
    if (url.includes('/stats/data-range')) return route.fulfill({ json: dataRangeFixture })
    if (url.includes('/stats/records')) return route.fulfill({ json: recordsFixture })
    if (url.includes('/stats/summary')) return route.fulfill({ json: summaryFixture })
    if (url.includes('/stats/experimental'))
        return route.fulfill({ json: experimentalStatsFixture })
    if (url.includes('/stats/trends')) return route.fulfill({ json: trendsFixture })
    if (url.includes('/stats/periods')) return route.fulfill({ json: periodsFixture })
    if (url.includes('/equipment/masks/epochs')) return route.fulfill({ json: maskEpochsFixture })
    if (url.includes('/equipment/masks')) return route.fulfill({ json: maskLogFixture })
    if (url.includes('/rx/all')) return route.fulfill({ json: rxAllFixture })
    if (url.includes('/rx/changes')) return route.fulfill({ json: rxChangesFixture })
    if (url.includes('/rx/compare')) return route.fulfill({ json: rxCompareFixture })
    if (url.includes('/rx/current')) return route.fulfill({ json: rxCurrentFixture })
    if (url.includes('/rx/history')) return route.fulfill({ json: rxHistoryFixture })
    if (url.includes('/devices/1')) return route.fulfill({ json: deviceDetailFixture })
    if (url.includes('/devices')) return route.fulfill({ json: devicesFixture })
    if (url.includes('/sessions/1470/events')) return route.fulfill({ json: sessionEventsFixture })
    if (url.includes('/sessions/1470/waveforms'))
        return route.fulfill({ json: flowWaveformFixture })
    if (url.includes('/sessions/1470')) return route.fulfill({ json: sessionDetailFixture })
    if (url.includes('/days/dates')) return route.fulfill({ json: dateListFixture })
    if (/\/days\/[^/]+\/experimental/.test(url))
        return route.fulfill({ json: dayExperimentalFixture })
    if (url.includes('/days')) return route.fulfill({ json: daysFixture })
    if (url.includes('/sessions')) return route.fulfill({ json: sessionsFixture })

    return route.fulfill({ json: {} })
}

// Flat layout: scripts/post-screenshots.sh globs ui/screenshots/*.png with
// -maxdepth 1, and desktop names predate the mobile project — keep them as-is.
function shotPath(testInfo: TestInfo, name: string) {
    const suffix = testInfo.project.name === 'mobile' ? '-mobile' : ''
    return `screenshots/${name}${suffix}.png`
}

// The Dark Mode toggle lives in the desktop sidebar, which is hidden at the
// mobile breakpoint. Mobile dark-mode coverage lives in 'dashboard dark (mobile)'.
function skipOnMobile(testInfo: TestInfo) {
    test.skip(testInfo.project.name === 'mobile', 'Dark Mode toggle is desktop-sidebar-only')
}

// Inverse gate for scenarios that only exist at the mobile breakpoint.
function skipOnDesktop(testInfo: TestInfo) {
    test.skip(testInfo.project.name !== 'mobile', 'Mobile-only layout scenario')
}

// Regression guard: mobile layouts must never scroll sideways. No-op on desktop
// so every test can call it unconditionally. Production CSS clips .app-main
// horizontally below 768px (App.vue), which stops overflow propagating to the
// document and would blind the document-level check — lift the clip before
// measuring and check both the document and .app-main itself. Callers must
// screenshot BEFORE calling this guard so captures keep production CSS.
async function expectNoHorizontalOverflow(page: Page, testInfo: TestInfo) {
    if (testInfo.project.name !== 'mobile') return
    await page.addStyleTag({ content: '.app-main { overflow-x: visible !important; }' })
    const { documentOverflows, mainOverflows } = await page.evaluate(() => {
        const main = document.querySelector('.app-main')
        return {
            documentOverflows: document.documentElement.scrollWidth > window.innerWidth,
            mainOverflows: main !== null && main.scrollWidth > main.clientWidth,
        }
    })
    expect(documentOverflows, 'document must not scroll sideways').toBe(false)
    expect(mainOverflows, '.app-main content must fit its width').toBe(false)
}

test.beforeEach(async ({ page }) => {
    await page.route('/api/v1/**', routeApi)
})

test('dashboard', async ({ page }, testInfo) => {
    await page.goto('/')
    await page.waitForSelector('.summary-row')
    await page.waitForTimeout(800)
    await page.screenshot({ path: shotPath(testInfo, 'dashboard') })
    await expectNoHorizontalOverflow(page, testInfo)
})

test('sessions', async ({ page }, testInfo) => {
    await page.goto('/sessions')
    // Below 768px the table is replaced by a card list, so wait on whichever renders.
    // Loading skeletons reuse .data-card and tbody tr markup, so wait on elements
    // that only exist once real data is in: the mobile sort row and table row links.
    if (testInfo.project.name === 'mobile') await page.waitForSelector('.mobile-sort-row')
    else await page.waitForSelector('.sessions-table tbody tr a')
    await page.screenshot({ path: shotPath(testInfo, 'sessions') })
    await expectNoHorizontalOverflow(page, testInfo)
})

test('stats', async ({ page }, testInfo) => {
    await page.goto('/stats')
    await page.waitForSelector('.records-grid')
    await page.waitForSelector('.trend-chart canvas')
    await page.waitForTimeout(800)
    await page.screenshot({ path: shotPath(testInfo, 'stats') })
    await expectNoHorizontalOverflow(page, testInfo)
})

test('stats error state', async ({ page }, testInfo) => {
    await page.route('**/api/v1/stats/**', (route) =>
        route.fulfill({
            status: 500,
            contentType: 'application/json',
            body: JSON.stringify({ detail: 'Internal Server Error' }),
        }),
    )
    await page.goto('/stats')
    await page.waitForSelector('.error-state')
    await page.waitForTimeout(800)
    await page.screenshot({ path: shotPath(testInfo, 'stats-error') })
    await expectNoHorizontalOverflow(page, testInfo)
})

test('rx-history', async ({ page }, testInfo) => {
    await page.goto('/rx')
    await page.waitForSelector('.setting-pill')
    await page.screenshot({ path: shotPath(testInfo, 'rx-history') })
    await expectNoHorizontalOverflow(page, testInfo)
})

test('session-detail', async ({ page }, testInfo) => {
    await page.goto('/sessions/1470')
    await page.waitForSelector('.session-detail .stats-section')
    await page.waitForSelector('canvas') // chart mounted
    await page.waitForTimeout(400) // let canvas paint settle
    await page.screenshot({ path: shotPath(testInfo, 'session-detail') })
    await expectNoHorizontalOverflow(page, testInfo)
})

test('reports', async ({ page }, testInfo) => {
    await page.goto('/reports')
    await page.waitForSelector('.reports-view')
    await page.getByText('Comparison').click()
    await page.waitForSelector('.rx-select')
    await page.screenshot({ path: shotPath(testInfo, 'reports') })
    await expectNoHorizontalOverflow(page, testInfo)
})

test('dashboard dark', async ({ page }, testInfo) => {
    skipOnMobile(testInfo)
    await page.goto('/')
    await page.waitForSelector('.summary-row')
    await page.waitForTimeout(800)
    await page.getByText('Dark Mode').click()
    await page.waitForTimeout(300)
    await page.screenshot({ path: shotPath(testInfo, 'dashboard-dark') })
})

test('session-detail dark', async ({ page }, testInfo) => {
    skipOnMobile(testInfo)
    await page.goto('/sessions/1470')
    await page.waitForSelector('.session-detail .stats-section')
    await page.waitForSelector('canvas') // chart mounted
    await page.waitForTimeout(400) // let canvas paint settle
    await page.getByText('Dark Mode').click()
    await page.waitForTimeout(300)
    await page.screenshot({ path: shotPath(testInfo, 'session-detail-dark') })
})

test('equipment', async ({ page }, testInfo) => {
    await page.goto('/equipment')
    await page.waitForSelector('.equipment-view .device-card')
    await page.waitForTimeout(300)
    await page.screenshot({ path: shotPath(testInfo, 'equipment') })
    await expectNoHorizontalOverflow(page, testInfo)
})

test('equipment dark', async ({ page }, testInfo) => {
    skipOnMobile(testInfo)
    await page.goto('/equipment')
    await page.waitForSelector('.equipment-view .device-card')
    await page.getByText('Dark Mode').click()
    await page.waitForTimeout(300)
    await page.screenshot({ path: shotPath(testInfo, 'equipment-dark') })
})

test('more-sheet', async ({ page }, testInfo) => {
    skipOnDesktop(testInfo)
    await page.goto('/')
    await page.waitForSelector('.summary-row')
    await page.getByRole('button', { name: 'More', exact: true }).click()
    await page.waitForSelector('[data-slot="sheet-content"]')
    await page.waitForTimeout(300) // let slide-in animation settle
    await page.screenshot({ path: shotPath(testInfo, 'more-sheet') })
    await expectNoHorizontalOverflow(page, testInfo)
})

test('sessions-cards', async ({ page }, testInfo) => {
    skipOnDesktop(testInfo)
    await page.goto('/sessions')
    // Loading skeletons reuse .card-list/.data-card markup — wait on the sort
    // row, which only renders once real session data has loaded.
    await page.waitForSelector('.mobile-sort-row')
    await expect(page.locator('.card-list')).toBeVisible()
    await expect(page.locator('.sessions-table')).toHaveCount(0)
    await page.screenshot({ path: shotPath(testInfo, 'sessions-cards') })
    await expectNoHorizontalOverflow(page, testInfo)
})

test('dashboard dark (mobile)', async ({ page }, testInfo) => {
    skipOnDesktop(testInfo)
    await page.goto('/')
    await page.waitForSelector('.summary-row')
    await page.waitForTimeout(800)
    await page.getByRole('button', { name: 'More', exact: true }).click()
    await page.waitForSelector('[data-slot="sheet-content"]')
    // Scope to the sheet: the hidden desktop sidebar has a matching toggle too.
    await page.locator('[data-slot="sheet-content"]').getByText('Dark Mode').click()
    await page.keyboard.press('Escape') // toggling does not navigate, so close manually
    await page.waitForSelector('[data-slot="sheet-content"]', { state: 'hidden' })
    await page.waitForTimeout(300)
    await page.screenshot({ path: shotPath(testInfo, 'dashboard-dark') })
    await expectNoHorizontalOverflow(page, testInfo)
})
