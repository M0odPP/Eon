# Eon Language Extension for VS Code / Cursor / VSCodium

Official zero-dependency editor extension for **Eon**:

- **Syntax highlighting** for Eon `v0.2` (`pool`, `view`, `linear`, `errdefer`, `@hal`, `@gpu`, `@stack(N)`, SIMD vectors, checked/wrapping operators).
- **Wing file icons** (`Eon Wing File Icons` theme) — white/dark wing for `.eon`, `build.eon`, and `eon.lock`, plus amber TCB boundary wing for `.hal.eon`.
- **Built-in `eonc lsp` integration** (zero `npm` dependencies required): live diagnostics, hover with DBM-proven ranges and `@stack(N)` frame sizes, go-to-definition, completion, and `eonc fmt` document formatting.

## Local Install

```bash
ln -s "$(pwd)/editors/vscode" "$HOME/.vscode/extensions/eon-lang"
```

Then restart your editor and select **File Icon Theme -> Eon Wing File Icons** (or use the per-language icon automatically). Set `eon.compilerPath` if `eonc` is not in `$PATH` or `selfhost/out/eonc2_rel`.
