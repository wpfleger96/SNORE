import { onBeforeUnmount, ref } from 'vue'

// Grace period so moving the pointer from the trigger into the portal-rendered
// content (across the gap between them) does not close the popover.
const CLOSE_DELAY_MS = 150

/**
 * Controlled open state for a Popover that also opens on mouse hover, so
 * click/tap toggling and hover cooperate (same behavior as InfoHint).
 *
 * Bind `open` with `v-model:open`, the trigger handlers to the trigger button,
 * and the content handlers to PopoverContent.
 */
export function useHoverPopover() {
    const open = ref(false)
    let closeTimer: ReturnType<typeof setTimeout> | null = null

    function cancelClose() {
        if (closeTimer !== null) {
            clearTimeout(closeTimer)
            closeTimer = null
        }
    }

    // Hover is mouse-only: touch taps fire pointerenter before click, so
    // guarding on pointerType keeps tap-to-toggle intact for touch users.
    function onTriggerPointerEnter(event: PointerEvent) {
        if (event.pointerType !== 'mouse') return
        cancelClose()
        open.value = true
    }

    function onPointerLeave(event: PointerEvent) {
        if (event.pointerType !== 'mouse') return
        cancelClose()
        closeTimer = setTimeout(() => {
            open.value = false
            closeTimer = null
        }, CLOSE_DELAY_MS)
    }

    // Never steal keyboard focus on open; focus stays on the trigger.
    function onOpenAutoFocus(event: Event) {
        event.preventDefault()
    }

    onBeforeUnmount(cancelClose)

    return { open, onTriggerPointerEnter, onPointerLeave, cancelClose, onOpenAutoFocus }
}
