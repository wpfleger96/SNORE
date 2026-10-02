import { describe, expect, it, vi, beforeEach } from 'vitest'
import { flushPromises, mount, type VueWrapper } from '@vue/test-utils'
import type { DayDetail } from '@/types'

vi.mock('@/api/days', () => ({
    getDay: vi.fn(),
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

import { getDay } from '@/api/days'
import DayDetailView from '@/views/DayDetailView.vue'

// A #299-shaped DayDetail: the real backend field names, including the
// null-reason companions (fl_class_ge4_pct_reason, rera_index_reason,
// rera_count_reason). Only the fields DayDetailView reads are populated.
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

describe('DayDetailView FL/RERA cards', () => {
    beforeEach(() => {
        vi.clearAllMocks()
    })

    it('test_experimental_cards_hidden_on_pre_299_shape', async () => {
        // Pre-#299 responses omit the fields entirely: value AND reason both absent.
        const wrapper = await mountDay(makeDay())

        expect(findCard(wrapper, 'FL Class ≥4')).toBeUndefined()
        expect(findCard(wrapper, 'RERA Proxy Index')).toBeUndefined()
        expect(findCard(wrapper, 'RERA Proxy Count')).toBeUndefined()
        expect(wrapper.find('[role="note"]').exists()).toBe(false)
    })

    it('test_experimental_cards_sit_under_one_experimental_banner', async () => {
        const wrapper = await mountDay(makeDay({ fl_class_ge4_pct: 3, rera_index: 5.5, reras: 2 }))

        const banners = wrapper.findAll('[role="note"]')
        expect(banners).toHaveLength(1)
        const group = banners[0]!.element.parentElement!
        expect(group.textContent).toContain('RERA Proxy Index')
        expect(group.textContent).toContain('FL Class ≥4')
        // The device-scored RERA count is not part of the experimental group.
        expect(group.textContent).not.toContain('RERAs')
    })

    it('test_null_value_with_reason_renders_em_dash_and_tooltip', async () => {
        const wrapper = await mountDay(
            makeDay({
                fl_class_ge4_pct: null,
                fl_class_ge4_pct_reason: 'analysis_not_run',
                rera_index: null,
                rera_index_reason: 'analysis_not_run',
                rera_count: null,
                rera_count_reason: 'analysis_not_run',
            }),
        )

        // The RERA Proxy Count card is the regression guard: reading the wrong
        // reason field (rera_reason) would hide it on exactly this path.
        const countCard = findCard(wrapper, 'RERA Proxy Count')
        expect(countCard).toBeDefined()
        const empty = countCard!.find('.stat-empty')
        expect(empty.text()).toBe('---')
        expect(empty.attributes('title')).toBe('Breath analysis has not been run for this night.')

        // Siblings behave identically so the count card is not a special case.
        expect(findCard(wrapper, 'FL Class ≥4')!.find('.stat-empty').text()).toBe('---')
        expect(findCard(wrapper, 'RERA Proxy Index')!.find('.stat-empty').text()).toBe('---')
    })

    it('test_numeric_values_render_including_zero', async () => {
        const wrapper = await mountDay(
            makeDay({
                fl_class_ge4_pct: 0,
                rera_index: 5.5,
                rera_count: 0,
            }),
        )

        // 0 is a real value, not absence: the card renders rather than showing an em-dash.
        const countCard = findCard(wrapper, 'RERA Proxy Count')
        expect(countCard).toBeDefined()
        expect(countCard!.find('.stat-empty').exists()).toBe(false)
        expect(countCard!.find('.stat-value').text()).toBe('0')

        expect(findCard(wrapper, 'FL Class ≥4')!.find('.stat-value').text()).toContain('0.0')
        expect(findCard(wrapper, 'RERA Proxy Index')!.find('.stat-value').text()).toBe('5.50')
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
