import { describe, expect, it, vi } from 'vitest'
import { mount, type VueWrapper } from '@vue/test-utils'
import type { DeviceChannelStats } from '@/types'

vi.mock('@/components/InfoHint.vue', () => ({
    default: { template: '<span class="info-hint-stub" />' },
}))

import DeviceChannelStatsTable from '@/components/DeviceChannelStatsTable.vue'

const FULL_STATS: DeviceChannelStats = {
    device_flg_median: 0.05,
    device_flg_95th: 0.214,
    device_flg_max: 0.68,
    snore_median: 0.1,
    snore_95th: 1.5,
    snore_max: 4.25,
}

function mountStats(stats: DeviceChannelStats | null | undefined): VueWrapper {
    return mount(DeviceChannelStatsTable, { props: { stats } })
}

function bodyRows(wrapper: VueWrapper): string[][] {
    return wrapper.findAll('tbody tr').map((row) => row.findAll('td').map((cell) => cell.text()))
}

describe('DeviceChannelStatsTable', () => {
    it('test_both_channels_render_median_95th_max_to_two_decimals', () => {
        expect(bodyRows(mountStats(FULL_STATS))).toEqual([
            ['Flow Limitation (device)', '0.05', '0.21', '0.68'],
            ['Snore (device)', '0.10', '1.50', '4.25'],
        ])
    })

    it('test_zero_values_render_as_real_measurements', () => {
        const wrapper = mountStats({
            ...FULL_STATS,
            device_flg_median: 0,
            device_flg_95th: 0,
            device_flg_max: 0,
        })

        expect(bodyRows(wrapper)[0]).toEqual(['Flow Limitation (device)', '0.00', '0.00', '0.00'])
    })

    it('test_missing_channel_hides_only_its_row', () => {
        const wrapper = mountStats({
            device_flg_median: null,
            device_flg_95th: null,
            device_flg_max: null,
            snore_median: 0,
            snore_95th: 1.5,
            snore_max: 4.25,
        })

        expect(bodyRows(wrapper)).toEqual([['Snore (device)', '0.00', '1.50', '4.25']])
    })

    it('test_partly_null_channel_keeps_its_row_with_a_placeholder', () => {
        const wrapper = mountStats({ snore_median: null, snore_95th: 1.5, snore_max: null })

        expect(bodyRows(wrapper)).toEqual([['Snore (device)', '---', '1.50', '---']])
    })

    it.each([
        ['null', null],
        ['undefined', undefined],
        ['all channels null', { device_flg_median: null, snore_max: null }],
    ])('test_section_hidden_when_stats_are_%s', (_, stats) => {
        expect(mountStats(stats).find('section').exists()).toBe(false)
    })
})
