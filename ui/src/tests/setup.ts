import { installMatchMediaMock } from './matchMedia'

// jsdom does not implement window.matchMedia, which @vueuse's useMediaQuery
// (used by useIsMobile) requires. Install the controllable mock before any
// component module imports useIsMobile (its ref binds at module scope), so
// tests can flip breakpoints deterministically via setMediaMatches().
// The typeof guard keeps this setup file inert in node-environment suites.
if (typeof window !== 'undefined') {
    installMatchMediaMock()
}
