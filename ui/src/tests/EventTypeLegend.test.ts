import { describe, expect, it } from 'vitest'
import { mount } from '@vue/test-utils'
import EventTypeLegend from '@/components/EventTypeLegend.vue'
import { EVENT_COLORS, EVENT_SOLID_COLORS } from '@/types'

describe('EventTypeLegend', () => {
    it('test_default_legend_includes_unclassified_apnea', () => {
        const wrapper = mount(EventTypeLegend)

        expect(wrapper.text()).toContain('UA — Unclassified Apnea')
    })

    it('test_every_default_entry_has_a_label_description_and_colour', () => {
        const wrapper = mount(EventTypeLegend)

        const rows = wrapper.findAll('li')
        expect(rows).toHaveLength(Object.keys(EVENT_COLORS).length)
        for (const row of rows) {
            expect(row.text()).toMatch(/^\S+ — .+ — .+/)
            expect(row.find('span').attributes('style')).toContain('background-color')
        }
    })
})

describe('event colours', () => {
    it.each([
        ['EVENT_COLORS', EVENT_COLORS],
        ['EVENT_SOLID_COLORS', EVENT_SOLID_COLORS],
    ])('test_%s_gives_each_event_type_a_distinct_hue', (_name, colors) => {
        const hues = Object.values(colors).map((c) => c.slice(0, c.lastIndexOf(',')))

        expect(new Set(hues).size).toBe(hues.length)
    })
})
