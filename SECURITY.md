# Security Policy

## Reporting a Vulnerability or Soundness Issue

Eon treats compiler soundness bugs (e.g., bypassing borrow checking, linear resource leaks, DBM bounds/overflow proof holes, `@hal`/`raw` boundary escapes, `@stack(N)` under-estimation, or `.emi` cache staleness) as high-severity issues.

Please report security vulnerabilities or soundness bypasses responsibly:

1. **GitHub Security Advisories:** Use the repository's **Security -> Report a vulnerability** form (preferred for sensitive findings).
2. **Public Soundness Issues:** For non-exploitable compiler soundness bugs, open a GitHub issue using the **Soundness Bug** template with:
   - A minimal self-contained `.eon` reproducer.
   - The compiler stage (`eonc0` vs `selfhost/out/eonc2_rel`) and flags (`--dev`, `--release`, `--sanitize`, `--emi-cache`, `-j N`).
   - Expected diagnostic (`E0xxx` / `ZM-xx`) vs actual behavior.

## Scope & Guarantees

- Safe `.eon` modules forbid `raw`, inline `asm`, `extern "C"`, and raw pointer dereferences. All unsafe / OS / hardware operations are isolated in `.hal.eon` modules and auditable via `eonc build --audit-tcb`.
- Eon is an actively developed systems language (pre-1.0). While verified by triple bootstrap, differential fuzzing, and soundness mutation fuzzing, it has not yet undergone a formal third-party security audit and should not yet be deployed in safety-critical production environments.
