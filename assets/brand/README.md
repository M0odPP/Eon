# Eon Brand Assets

Official vector marks and artwork for the **Eon** programming language.

| Asset | File | Usage |
| :--- | :--- | :--- |
| **Wing Mark (White)** | [eon-mark-white.svg](eon-mark-white.svg) | Dark backgrounds, terminal/IDE dark themes, social cards |
| **Wing Mark (Dark)** | [eon-mark-dark.svg](eon-mark-dark.svg) | Light backgrounds, documentation, light GitHub theme |
| **Wing Mark (`currentColor`)** | [eon-mark.svg](eon-mark.svg) | Inline web icons, adaptive UI themes, favicons |
| **App Icon (Squircle)** | [eon-app-icon.svg](eon-app-icon.svg) | OS app bundles (`.desktop`, `.app`, APK, Windows), VS Code extension |
| **Wordmark (Dark / Light)** | [eon-wordmark-dark.svg](eon-wordmark-dark.svg) / [eon-wordmark-light.svg](eon-wordmark-light.svg) | Repository header, release announcements, web hero |
| **GitHub Social Preview** | [eon-social-preview.svg](eon-social-preview.svg) | `1280×640` repository OpenGraph preview |

## Geometry & Colors

The Eon wing mark is defined in a `128×128` coordinate system by two closed cubic-Bezier paths:

- **Primary wing:** `M 18 82 C 42 54 70 34 108 24 C 86 52 68 70 50 84 L 56 64 C 42 70 30 76 18 82 Z`
- **Secondary wing:** `M 57 88 C 72 76 86 62 100 44 C 92 70 80 90 62 106 L 57 88 Z`

The exact same Bezier geometry is embedded directly in `libs/eon_ui/src/brand.eon` and `libs/eon_win/src/icon.eon` for crisp HiDPI native GPU vector rendering and OS window icons without external raster assets.

- **Light Foreground (on Dark):** `#FFFFFF`
- **Dark Foreground (on Light):** `#0F172A`
- **App Icon Surface:** `#111827 -> #090D16` with `#334155 -> #1E293B` border
- **HAL Boundary Accent (`.hal.eon` icons):** `#F59E0B` (dark theme) / `#D97706` (light theme)
