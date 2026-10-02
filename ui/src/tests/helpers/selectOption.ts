import { flushPromises, type VueWrapper } from '@vue/test-utils'
import { vi } from 'vitest'

/** Stub the DOM APIs reka-ui's Select calls that jsdom does not implement. */
export function installSelectDomStubs(): void {
    Element.prototype.scrollIntoView = vi.fn()
    Element.prototype.hasPointerCapture = vi.fn(() => false)
    Element.prototype.releasePointerCapture = vi.fn()
}

/** Open the real reka-ui Select at `trigger` and choose the option labelled `label`.
 *  The wrapper must be mounted with `attachTo: document.body` (content is portalled). */
export async function chooseSelectOption(
    wrapper: VueWrapper,
    label: string,
    trigger = '[role="combobox"]',
): Promise<void> {
    wrapper
        .get(trigger)
        .element.dispatchEvent(
            new PointerEvent('pointerdown', { bubbles: true, button: 0, pointerType: 'mouse' }),
        )
    await flushPromises()
    const option = Array.from(document.body.querySelectorAll('[role="option"]')).find(
        (el) => el.textContent?.trim() === label,
    )
    if (!option) throw new Error(`Select option "${label}" not rendered`)
    option.dispatchEvent(new PointerEvent('pointerup', { bubbles: true, pointerType: 'mouse' }))
    await flushPromises()
}
