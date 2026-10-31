import { describe, expect, it, vi, beforeEach } from 'vitest'
import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import type { DayDetail } from '@/types'

vi.mock('@/api/days', () => ({
    getDay: vi.fn(),
    getDayExperimental: vi.fn(),
}))

// Stub InfoHint so stat-card labels render as plain text (no Popover/teleport),
// keeping the real StatCard so its em-dash + title-tooltip logic is exercised.
vi.mock('@/components/InfoHint.vue', () => ({
    default: { template: '<span class="info-hint-stub" />' },
}))

vi.mock('@lucide/vue', () => ({
    Loader2: { template: '<svg class="loader-stub" />' },
    ArrowLeft: { template: '<svg class="arrow-stub" />' },
    FlaskConical: { template: '<svg class="flask-stub" />' },
}))

import { getDay, getDayExperimental } from '@/api/days'
import DayDetailView from '@/views/DayDetailView.vue'

// Only the fields DayDetailView reads are populated.
function makeDay(overrides: Record<string, unknown> = {}): DayDetail {
    return {
        date: '2026-05-03',
        session_count: 1,
        ...overrides,
    } as unknown as DayDetail
}

async function mountDay(day: DayDetail): Promise<VueWrapper> {
    vi.mocked(getDay).mockResolvedValue(day)
    const wrapper = mount(DayDetailView, {
        props: { dayDate: '2026-05-03' },
        global: { stubs: { RouterLink: { template: '<a><slot /></a>' } } },
    })
    await flushPromises()
    return wrapper
}

/** Locate a stat card by its (exact) label text; undefined when the card is not rendered. */
function findCard(wrapper: VueWrapper, label: string) {
    return wrapper.findAll('.stat-card').find((c) => c.find('.stat-label').text() === label)
}

describe('DayDetailView experimental gating', () => {
    beforeEach(() => {
        vi.clearAllMocks()
    })

    it('test_default_page_never_requests_experimental_metrics', async () => {
        const wrapper = await mountDay(makeDay())

        expect(getDayExperimental).not.toHaveBeenCalled()
        expect(findCard(wrapper, 'FL Class ≥4')).toBeUndefined()
        expect(findCard(wrapper, 'RERA Proxy Index')).toBeUndefined()
        expect(findCard(wrapper, 'RERA Proxy Count')).toBeUndefined()
        expect(wrapper.find('[role="note"]').exists()).toBe(false)
    })

    it('test_experimental_link_targets_the_days_experimental_page', async () => {
        const wrapper = await mountDay(makeDay())

        expect(wrapper.find('a[to="/days/2026-05-03/experimental"]').exists()).toBe(true)
    })
})

describe('DayDetailView AHI source note', () => {
    beforeEach(() => {
        vi.clearAllMocks()
    })

    function ahiNote(wrapper: VueWrapper): string | null {
        const note = findCard(wrapper, 'AHI')!.find('.stat-footnote')
        return note.exists() ? note.text() : null
    }

    it('test_device_headline_names_source_and_shows_recount', async () => {
        const wrapper = await mountDay(
            makeDay({ ahi: 2.1, ahi_computed: 2.64, index_source: 'device' }),
        )
        expect(ahiNote(wrapper)).toBe('Device-reported; SNORE recount 2.6')
    })

    it('test_derived_headline_is_labelled_snore_recount', async () => {
        const wrapper = await mountDay(
            makeDay({ ahi: 2.64, ahi_computed: 2.64, index_source: 'derived' }),
        )
        expect(ahiNote(wrapper)).toBe('SNORE recount')
    })

    it('test_unknown_source_or_missing_ahi_shows_no_note', async () => {
        expect(ahiNote(await mountDay(makeDay({ ahi: 2.1, index_source: null })))).toBeNull()
        expect(ahiNote(await mountDay(makeDay({ ahi: null, index_source: 'device' })))).toBeNull()
    })
})
