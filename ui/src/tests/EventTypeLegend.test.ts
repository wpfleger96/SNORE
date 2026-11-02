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
        // CAA shares CA's colour and reads as central apnea, so it has no legend row of its own.
        expect(rows).toHaveLength(Object.keys(EVENT_COLORS).length - 1)
        expect(wrapper.text()).not.toContain('CAA')
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
        const { CAA, ...distinctTypes } = colors
        expect(CAA).toBe(colors.CA)
        const hues = Object.values(distinctTypes).map((c) => c.slice(0, c.lastIndexOf(',')))

        expect(new Set(hues).size).toBe(hues.length)
    })
})
