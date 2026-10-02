import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { flushPromises, mount } from '@vue/test-utils'
import { ref } from 'vue'
import { createRouter, createMemoryHistory } from 'vue-router'

vi.mock('@/composables/useAuth')
vi.mock('@/api/me', () => ({ getMe: vi.fn() }))

import App from '@/App.vue'
import AppSidebar from '@/components/AppSidebar.vue'
import { useAuth } from '@/composables/useAuth'
import { makeAuthMock } from './helpers/mockUseAuth'
import { setMediaMatches } from './matchMedia'

function makeRouter() {
    return createRouter({
        history: createMemoryHistory(),
        routes: [{ path: '/:pathMatch(.*)*', component: { template: '<div />' } }],
    })
}

function findLegendTrigger(root: ParentNode): HTMLButtonElement | undefined {
    return [...root.querySelectorAll('button')].find((b) =>
        b.textContent?.includes('Data provenance'),
    )
}

describe('provenance legend in navigation', () => {
    beforeEach(() => {
        vi.mocked(useAuth).mockReturnValue(
            makeAuthMock({ isAuthenticated: ref(true), isLocal: ref(true) }) as never,
        )
    })

    afterEach(() => {
        setMediaMatches(false)
        document.body.innerHTML = ''
    })

    it('test_sidebar_footer_legend_expands_to_tier_key', async () => {
        const router = makeRouter()
        await router.push('/dashboard')
        const wrapper = mount(AppSidebar, {
            global: { plugins: [router] },
            attachTo: document.body,
        })

        const trigger = findLegendTrigger(wrapper.element)
        expect(trigger).toBeDefined()
        expect(wrapper.find('[aria-label="Data provenance legend"]').exists()).toBe(false)

        trigger!.click()
        await flushPromises()

        const legend = wrapper.find('[aria-label="Data provenance legend"]')
        expect(legend.exists()).toBe(true)
        expect(legend.text()).toContain('Derived')
        expect(legend.text()).toContain('Experimental')
        wrapper.unmount()
    })

    it('test_mobile_nav_sheet_includes_legend', async () => {
        setMediaMatches(true)
        const router = makeRouter()
        await router.push('/dashboard')
        const wrapper = mount(App, { global: { plugins: [router] }, attachTo: document.body })

        const more = wrapper.findAll('.mobile-tab-bar button').find((b) => b.text() === 'More')
        await more!.trigger('click')
        await flushPromises()

        const sheet = document.body.querySelector('[role="dialog"]')
        expect(sheet).not.toBeNull()
        const trigger = findLegendTrigger(sheet!)
        expect(trigger).toBeDefined()

        trigger!.click()
        await flushPromises()

        expect(sheet!.querySelector('[aria-label="Data provenance legend"]')).not.toBeNull()
        wrapper.unmount()
    })
})
