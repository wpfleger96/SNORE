import { afterEach } from 'vitest'
import { installMatchMediaMock } from './matchMedia'

// jsdom does not implement window.matchMedia, which @vueuse's useMediaQuery
// (used by useIsMobile) requires. Install the controllable mock before any
// component module imports useIsMobile (its ref binds at module scope), so
// tests can flip breakpoints deterministically via setMediaMatches().
// The typeof guard keeps this setup file inert in node-environment suites.
if (typeof window !== 'undefined') {
    installMatchMediaMock()
}

// A provenance lookup warning means a mark silently resolved to a fallback tier
// (missing schema, untagged field, unknown source). Fail the test that caused
// it, or the first test of a file whose module-level code did. Tests that
// assert on the warning replace console.warn with vi.spyOn and bypass this.
const PROVENANCE_WARNING = /^\[(provenanceFor|glossaryProvenance)\]/
let provenanceWarnings: string[] = []
const originalWarn = console.warn
console.warn = (...args: unknown[]) => {
    if (typeof args[0] === 'string' && PROVENANCE_WARNING.test(args[0])) {
        provenanceWarnings.push(args[0])
    }
    originalWarn(...args)
}
afterEach(() => {
    const warnings = provenanceWarnings
    provenanceWarnings = []
    if (warnings.length) throw new Error(`Provenance lookup warnings:\n${warnings.join('\n')}`)
})
