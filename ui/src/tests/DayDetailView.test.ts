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
    AlertTriangle: { template: '<svg class="alert-stub" />' },
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

function mountView(): VueWrapper {
    return mount(DayDetailView, {
        props: { dayDate: '2026-05-03' },
        global: { stubs: { RouterLink: { template: '<a><slot /></a>' } } },
    })
}

async function mountDay(day: DayDetail): Promise<VueWrapper> {
    vi.mocked(getDay).mockResolvedValue(day)
    const wrapper = mountView()
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

describe('DayDetailView device event cards', () => {
    beforeEach(() => {
        vi.clearAllMocks()
    })

    it.each([
        ['RERAs', 'reras'],
        ['Unclassified Apneas', 'unclassified_apneas'],
    ])('test_%s_card_hidden_on_null_and_shows_real_zero', async (label, field) => {
        expect(findCard(await mountDay(makeDay({ [field]: null })), label)).toBeUndefined()

        const card = findCard(await mountDay(makeDay({ [field]: 0 })), label)
        expect(card?.find('.stat-value').text()).toBe('0')
    })

    it('test_unclassified_apneas_card_follows_central_apneas', async () => {
        const wrapper = await mountDay(makeDay({ central_apneas: 1, unclassified_apneas: 2 }))

        const labels = wrapper.findAll('.stat-label').map((l) => l.text())
        expect(labels[labels.indexOf('Central Apneas') + 1]).toBe('Unclassified Apneas')
    })
})

describe('DayDetailView device channel and mask-on sections', () => {
    const CHANNEL_STATS = { snore_median: 0, snore_95th: 0.4, snore_max: 1.2 }
    const SPANS = [
        {
            session_id: 7,
            start: '2026-05-03T22:00:00',
            end: '2026-05-04T06:00:00',
            duration_seconds: 28800,
        },
    ]

    beforeEach(() => {
        vi.clearAllMocks()
    })

    function sectionHeadings(wrapper: VueWrapper): string[] {
        return wrapper.findAll('section h2').map((h) => h.text())
    }

    it('test_both_sections_render_from_the_day_payload', async () => {
        const wrapper = await mountDay(
            makeDay({ channel_stats: CHANNEL_STATS, mask_on_spans: SPANS }),
        )

        expect(sectionHeadings(wrapper)).toEqual(['Flow Limitation & Snore', 'Mask-On Spans'])
    })

    it('test_unknown_channels_and_empty_spans_hide_both_sections', async () => {
        const wrapper = await mountDay(makeDay({ channel_stats: null, mask_on_spans: [] }))

        expect(sectionHeadings(wrapper)).toEqual([])
    })

    it('test_null_mask_on_spans_hides_the_section_without_error', async () => {
        const wrapper = await mountDay(
            makeDay({
                channel_stats: CHANNEL_STATS,
                mask_on_spans: null,
            }),
        )

        expect(sectionHeadings(wrapper)).toEqual(['Flow Limitation & Snore'])
    })
})

describe('DayDetailView date changes', () => {
    const NEXT_DATE = '2026-05-04'
    const notFound = Object.assign(new Error('Not Found'), {
        isAxiosError: true,
        response: { status: 404 },
    })

    beforeEach(() => {
        vi.clearAllMocks()
    })

    it('test_date_change_refetches_and_replaces_the_shown_day', async () => {
        const wrapper = await mountDay(makeDay({ total_therapy_hours: 7 }))
        vi.mocked(getDay).mockResolvedValue(makeDay({ date: NEXT_DATE, total_therapy_hours: 3 }))

        await wrapper.setProps({ dayDate: NEXT_DATE })
        await flushPromises()

        expect(getDay).toHaveBeenLastCalledWith(NEXT_DATE)
        expect(findCard(wrapper, 'Total Hours')?.text()).toContain('3.0')
    })

    it('test_slow_response_for_the_old_date_never_replaces_the_new_one', async () => {
        let resolveOld: (day: DayDetail) => void = () => {}
        vi.mocked(getDay)
            .mockReturnValueOnce(new Promise((resolve) => (resolveOld = resolve)))
            .mockResolvedValueOnce(makeDay({ date: NEXT_DATE, total_therapy_hours: 3 }))
        const wrapper = mountView()

        await wrapper.setProps({ dayDate: NEXT_DATE })
        await flushPromises()
        resolveOld(makeDay({ total_therapy_hours: 7 }))
        await flushPromises()

        expect(findCard(wrapper, 'Total Hours')?.text()).toContain('3.0')
    })

    it('test_date_with_no_data_replaces_the_shown_day_with_the_empty_state', async () => {
        const wrapper = await mountDay(makeDay({ total_therapy_hours: 7 }))
        vi.mocked(getDay).mockRejectedValue(notFound)

        await wrapper.setProps({ dayDate: NEXT_DATE })
        await flushPromises()

        expect(wrapper.text()).toContain('No CPAP data recorded for this date.')
        expect(findCard(wrapper, 'Total Hours')).toBeUndefined()
        expect(wrapper.find('.error-state').exists()).toBe(false)
    })

    it('test_other_failures_show_the_error_state', async () => {
        vi.mocked(getDay).mockRejectedValue(new Error('boom'))
        const wrapper = mountView()
        await flushPromises()

        expect(wrapper.find('.error-state').text()).toContain('boom')
        expect(wrapper.text()).not.toContain('No CPAP data recorded')
    })
})
