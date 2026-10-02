import { describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { ref } from 'vue'

const RUNS = [2, 1].map((run_id) => ({
    run_id,
    validator_type: 'rera',
    state: 'succeeded',
    date_from: '2026-01-01',
    date_to: '2026-01-31',
    created_at: '2026-02-01T00:00:00Z',
    engine_identity: {},
    validator_params: {},
}))

vi.mock('@/composables/useValidationRuns', () => ({
    useValidationRuns: () => ({
        runs: ref(RUNS),
        getDetail: vi.fn().mockResolvedValue({
            report_json: {
                aggregate: { mean_proxy_sensitivity: 0.2, machine_re_density: 0.5 },
            },
        }),
    }),
}))

import RunComparison from '@/components/validation/RunComparison.vue'

function markLabel(cell: Element): string | null {
    return cell.querySelector('button.provenance-mark')?.getAttribute('aria-label') ?? null
}

describe('RunComparison provenance marks', () => {
    it('test_metric_rows_render_tier_from_metric_config', async () => {
        const wrapper = mount(RunComparison)
        await flushPromises()

        const labelCell = (text: string) =>
            wrapper.findAll('td').find((td) => td.text().startsWith(text))!.element

        expect(markLabel(labelCell('Proxy sensitivity'))).toMatch(/^Experimental/)
        expect(markLabel(labelCell('Machine RE density'))).toMatch(/^Derived/)
    })
})
