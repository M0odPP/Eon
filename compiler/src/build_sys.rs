//! Sandboxed `build.eon` manifest execution and path validation.
//!
//! Compiles `build.eon` strictly under [`FileProfile::Safe`] (forbidding `@hal`,
//! `raw { ... }`, `asm(...)`, and `extern "C"` declarations), verifies the
//! `build`/`main` entry point accepts `*mut BuildCap`, runs the script inside the
//! EIR interpreter, and validates all resulting output and target paths so they
//! cannot escape `pkg_root`.

use std::collections::BTreeSet;
use std::ffi::OsString;
use std::fmt::{self, Write as _};
use std::fs;
use std::path::{Component, Path, PathBuf};

use crate::ast::CompilationUnit;
use crate::diag::DiagBag;
use crate::resolve::ResolvedProgram;
use crate::span::{FileProfile, Interner, SourceMap};
use crate::tcb_audit::scan_unit_constructs;
use crate::types::{BorrowInner, Ty, TypeEnv};

/// Output artifact kind produced by a `build.eon` target.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BuildTargetKind {
    Exe,
    Shared,
}

impl fmt::Display for BuildTargetKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Exe => f.write_str("exe"),
            Self::Shared => f.write_str("shared"),
        }
    }
}

/// Single build target declared via `BuildCap` inside `build.eon`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BuildTarget {
    pub name: String,
    pub root: String,
    pub kind: BuildTargetKind,
}

/// Validated declarative build plan produced by executing `build.eon`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BuildPlan {
    pub out_dir: String,
    pub sanitize: bool,
    pub audit_tcb: bool,
    pub needed_libs: Vec<String>,
    pub allow_hal: BTreeSet<String>,
    pub targets: Vec<BuildTarget>,
}

/// Validates that `rel` is a sandbox-safe relative path staying within `pkg_root`.
///
/// Rejects empty paths, NUL bytes, absolute/prefixed paths, `..` traversal,
/// dangling symlinks, and any existing ancestor resolving outside `canonical(pkg_root)`.
pub fn validate_sandbox_rel_path(pkg_root: &Path, rel: &str) -> Result<PathBuf, String> {
    if rel.is_empty() {
        return Err("build.eon sandbox error: relative path cannot be empty".to_owned());
    }
    if rel.contains('\0') {
        return Err("build.eon sandbox error: path contains NUL byte".to_owned());
    }
    if rel.starts_with('/') || rel.starts_with('\\') {
        return Err(format!(
            "build.eon sandbox error: absolute path `{rel}` is forbidden"
        ));
    }

    let rel_path = Path::new(rel);
    if rel_path.is_absolute() {
        return Err(format!(
            "build.eon sandbox error: absolute path `{rel}` is forbidden"
        ));
    }

    for comp in rel_path.components() {
        match comp {
            Component::Normal(_) | Component::CurDir => {}
            Component::ParentDir => {
                return Err(format!(
                    "build.eon sandbox error: path `{rel}` escapes package root via `..`"
                ));
            }
            Component::RootDir | Component::Prefix(_) => {
                return Err(format!(
                    "build.eon sandbox error: non-relative path `{rel}` is forbidden"
                ));
            }
        }
    }

    let joined = pkg_root.join(rel_path);
    let Ok(canon_root) = fs::canonicalize(pkg_root) else {
        return Ok(joined);
    };

    let mut probe = joined.as_path();
    let mut tail: Vec<OsString> = Vec::new();
    let canon_ancestor = loop {
        match fs::canonicalize(probe) {
            Ok(c) => break c,
            Err(_) => {
                if let Ok(meta) = fs::symlink_metadata(probe)
                    && meta.file_type().is_symlink()
                {
                    return Err(format!(
                        "build.eon sandbox error: path `{rel}` traverses unresolvable symlink `{}`",
                        probe.display()
                    ));
                }
                let Some(name) = probe.file_name() else {
                    return Err(format!(
                        "build.eon sandbox error: path `{rel}` has no resolvable ancestor under `{}`",
                        pkg_root.display()
                    ));
                };
                tail.push(name.to_os_string());
                probe = probe.parent().unwrap_or_else(|| Path::new(""));
            }
        }
    };

    if !canon_ancestor.starts_with(&canon_root) {
        return Err(format!(
            "build.eon sandbox error: path `{rel}` escapes package root `{}` (resolved to `{}`)",
            canon_root.display(),
            canon_ancestor.display()
        ));
    }

    let mut resolved = canon_ancestor;
    for seg in tail.into_iter().rev() {
        resolved.push(seg);
    }
    Ok(resolved)
}

/// Verifies that a parsed `build.eon` unit strictly obeys the [`FileProfile::Safe`] sandbox.
pub fn validate_build_unit_profile(
    build_file: &Path,
    unit: &CompilationUnit,
    sm: &SourceMap,
    interner: &Interner,
) -> Result<(), String> {
    let path_str = build_file.to_string_lossy();
    if path_str.ends_with(".hal.eon")
        || FileProfile::from_path(&path_str) == FileProfile::Hal
        || sm.file(unit.file).profile == FileProfile::Hal
        || unit.is_hal
    {
        return Err(
            "build.eon sandbox error: @hal/raw/asm is forbidden in build scripts".to_owned(),
        );
    }

    let scan = scan_unit_constructs(unit, interner);
    if !scan.raw_spans.is_empty() || scan.asm_sites > 0 || scan.has_hal_attr {
        return Err(
            "build.eon sandbox error: @hal/raw/asm is forbidden in build scripts".to_owned(),
        );
    }
    if !scan.extern_c_imports.is_empty() {
        return Err(
            "build.eon sandbox error: extern C/FFI declarations are forbidden in build scripts"
                .to_owned(),
        );
    }
    Ok(())
}

/// Formats a deterministic human-readable summary of a [`BuildPlan`].
pub fn format_build_summary(plan: &BuildPlan) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "EON BUILD PLAN");
    let _ = writeln!(out, "  out_dir: {}", plan.out_dir);
    let _ = writeln!(out, "  sanitize: {}", plan.sanitize);
    let _ = writeln!(out, "  audit_tcb: {}", plan.audit_tcb);
    if plan.needed_libs.is_empty() {
        let _ = writeln!(out, "  needed_libs: none");
    } else {
        let _ = writeln!(out, "  needed_libs: [{}]", plan.needed_libs.join(", "));
    }
    if plan.allow_hal.is_empty() {
        let _ = writeln!(out, "  allow_hal: none");
    } else {
        let allowed: Vec<&str> = plan.allow_hal.iter().map(String::as_str).collect();
        let _ = writeln!(out, "  allow_hal: [{}]", allowed.join(", "));
    }
    let _ = writeln!(out, "  targets: {}", plan.targets.len());
    for t in &plan.targets {
        let _ = writeln!(out, "    - {} ({}) -> {}", t.name, t.kind, t.root);
    }
    out
}

fn validate_build_entrypoint(
    resolved: &ResolvedProgram,
    env: &TypeEnv,
    interner: &Interner,
) -> Result<(), String> {
    let entry_sig = env
        .fns
        .values()
        .find(|f| interner.get(f.name) == "build")
        .or_else(|| env.fns.values().find(|f| interner.get(f.name) == "main"));

    let Some(sig) = entry_sig else {
        return Ok(());
    };

    let fn_name = interner.get(sig.name);
    if !sig.generics.is_empty() {
        return Err(format!(
            "build.eon sandbox error: entry function `{fn_name}` cannot be generic"
        ));
    }

    let valid_param = sig.params.len() == 1
        && matches!(
            env.get(sig.params[0].ty),
            Ty::Borrow {
                is_mut: true,
                inner: BorrowInner::Single(inner_ty),
            } if env.format_ty_named(*inner_ty, resolved, interner) == "BuildCap"
        );

    if !valid_param {
        return Err(format!(
            "build.eon sandbox error: entry function `{fn_name}` must take `(*mut BuildCap)`"
        ));
    }
    Ok(())
}

fn validate_target_name(name: &str) -> Result<(), String> {
    if name.is_empty()
        || name == "."
        || name == ".."
        || name.contains('\0')
        || name.contains('/')
        || name.contains('\\')
    {
        return Err(format!(
            "build.eon sandbox error: invalid target name `{name}`"
        ));
    }
    Ok(())
}

/// Compiles, verifies, and executes `build_file` inside the `build.eon` sandbox,
/// returning a path-validated [`BuildPlan`] anchored under `pkg_root`.
pub fn execute_build_eon(pkg_root: &Path, build_file: &Path) -> Result<BuildPlan, String> {
    let build_path_str = build_file.to_string_lossy().into_owned();
    if build_path_str.ends_with(".hal.eon")
        || FileProfile::from_path(&build_path_str) == FileProfile::Hal
    {
        return Err(
            "build.eon sandbox error: @hal/raw/asm is forbidden in build scripts".to_owned(),
        );
    }

    let bytes = fs::read(build_file)
        .map_err(|e| format!("failed to read build script `{}`: {e}", build_file.display()))?;

    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let mut diags = DiagBag::new();
    let checked = crate::check_files_in_mem(
        &[(build_path_str, bytes)],
        &mut sm,
        &mut interner,
        &mut diags,
    );

    if let Some(unit) = checked.units.first() {
        validate_build_unit_profile(build_file, unit, &sm, &interner)?;
    }
    if diags.has_errors() {
        return Err(diags.render_all(&sm));
    }

    validate_build_entrypoint(&checked.resolved, &checked.env, &interner)?;

    let mut plan = crate::eir_interp::interpret_build_script(
        &checked.eir,
        &checked.resolved,
        &checked.env,
        &interner,
    )?;

    if plan.out_dir.is_empty() {
        plan.out_dir = "out".to_owned();
    }
    validate_sandbox_rel_path(pkg_root, &plan.out_dir)?;

    for target in &plan.targets {
        validate_target_name(&target.name)?;
        validate_sandbox_rel_path(pkg_root, &target.root)?;
    }

    Ok(plan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::lex_file;
    use crate::parser::parse_file;

    fn unique_test_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "eon_build_sys_{}_{}",
            tag,
            std::process::id()
        ));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create test temp dir");
        dir
    }

    #[test]
    fn validate_sandbox_rel_path_rejects_escapes_and_accepts_safe_paths() {
        let pkg = unique_test_dir("sandbox");

        assert!(validate_sandbox_rel_path(&pkg, "/etc/passwd").is_err());
        assert!(validate_sandbox_rel_path(&pkg, "../escape").is_err());
        assert!(validate_sandbox_rel_path(&pkg, "src/../../escape").is_err());
        assert!(validate_sandbox_rel_path(&pkg, "").is_err());
        assert!(validate_sandbox_rel_path(&pkg, "bad\0path").is_err());
        assert!(validate_sandbox_rel_path(&pkg, "\\windows\\style").is_err());

        let ok_out = validate_sandbox_rel_path(&pkg, "out").expect("relative out dir allowed");
        assert!(ok_out.ends_with("out"));
        let ok_src =
            validate_sandbox_rel_path(&pkg, "./src/main.eon").expect("relative file allowed");
        assert!(ok_src.ends_with("main.eon"));

        #[cfg(unix)]
        {
            use std::os::unix::fs::symlink;
            let link_path = pkg.join("escape_link");
            symlink("/etc", &link_path).expect("create symlink");
            assert!(validate_sandbox_rel_path(&pkg, "escape_link/passwd").is_err());
        }

        let _ = fs::remove_dir_all(&pkg);
    }

    #[test]
    fn validate_build_unit_profile_blocks_hal_raw_asm_and_extern() {
        let cases = [
            (
                "build.hal.eon",
                "module build @hal;\npub fn helper() {}\n",
                "@hal/raw/asm is forbidden",
            ),
            (
                "build.eon",
                "module build @hal;\npub fn helper() {}\n",
                "@hal/raw/asm is forbidden",
            ),
            (
                "build.eon",
                "module build;\npub fn helper() { raw { let _x = 1_i32; } }\n",
                "@hal/raw/asm is forbidden",
            ),
            (
                "build.eon",
                "module build;\npub fn helper() { asm(\"nop\"); }\n",
                "@hal/raw/asm is forbidden",
            ),
            (
                "build.eon",
                "module build;\nextern \"C\" fn system(cmd: *u8) -> i32;\n",
                "extern C/FFI declarations are forbidden",
            ),
        ];

        for (name, src, expected_substr) in cases {
            let mut sm = SourceMap::new();
            let mut interner = Interner::new();
            let mut diags = DiagBag::new();
            let fid = sm.add_file_str(name, src);
            let file = sm.file(fid);
            let lexed = lex_file(file, &mut diags);
            let unit = parse_file(file, &lexed, &mut interner, &mut diags);
            let err = validate_build_unit_profile(Path::new(name), &unit, &sm, &interner)
                .expect_err("unsafe build script must be rejected");
            assert!(
                err.contains(expected_substr),
                "expected `{expected_substr}` in error `{err}`"
            );
        }

        let mut sm = SourceMap::new();
        let mut interner = Interner::new();
        let mut diags = DiagBag::new();
        let fid = sm.add_file_str("build.eon", "module build;\npub fn helper() {}\n");
        let file = sm.file(fid);
        let lexed = lex_file(file, &mut diags);
        let unit = parse_file(file, &lexed, &mut interner, &mut diags);
        assert!(validate_build_unit_profile(Path::new("build.eon"), &unit, &sm, &interner).is_ok());
    }

    #[test]
    fn execute_build_eon_rejects_hal_raw_and_asm_files() {
        let pkg = unique_test_dir("exec_reject");

        let hal_path = pkg.join("build.hal.eon");
        fs::write(&hal_path, "module build @hal;\n").expect("write hal build file");
        let err_hal_file =
            execute_build_eon(&pkg, &hal_path).expect_err(".hal.eon build script must fail");
        assert_eq!(
            err_hal_file,
            "build.eon sandbox error: @hal/raw/asm is forbidden in build scripts"
        );

        let raw_path = pkg.join("build.eon");
        fs::write(
            &raw_path,
            "module build;\npub fn main() { raw { let _x = 1_i32; } }\n",
        )
        .expect("write raw build file");
        let err_raw =
            execute_build_eon(&pkg, &raw_path).expect_err("raw {} in build.eon must fail");
        assert_eq!(
            err_raw,
            "build.eon sandbox error: @hal/raw/asm is forbidden in build scripts"
        );

        fs::write(&raw_path, "module build;\npub fn main() { asm(\"nop\"); }\n")
            .expect("write asm build file");
        let err_asm = execute_build_eon(&pkg, &raw_path).expect_err("asm in build.eon must fail");
        assert_eq!(
            err_asm,
            "build.eon sandbox error: @hal/raw/asm is forbidden in build scripts"
        );

        let _ = fs::remove_dir_all(&pkg);
    }

    #[test]
    fn format_build_summary_lists_fields_and_targets() {
        let mut allow_hal = BTreeSet::new();
        allow_hal.insert("drv.uart".to_owned());
        let plan = BuildPlan {
            out_dir: "out".to_owned(),
            sanitize: true,
            audit_tcb: true,
            needed_libs: vec!["c".to_owned()],
            allow_hal,
            targets: vec![
                BuildTarget {
                    name: "app".to_owned(),
                    root: "src/main.eon".to_owned(),
                    kind: BuildTargetKind::Exe,
                },
                BuildTarget {
                    name: "plugin".to_owned(),
                    root: "src/plugin.eon".to_owned(),
                    kind: BuildTargetKind::Shared,
                },
            ],
        };
        let summary = format_build_summary(&plan);
        assert!(summary.contains("out_dir: out"));
        assert!(summary.contains("sanitize: true"));
        assert!(summary.contains("audit_tcb: true"));
        assert!(summary.contains("drv.uart"));
        assert!(summary.contains("app (exe) -> src/main.eon"));
        assert!(summary.contains("plugin (shared) -> src/plugin.eon"));
    }
}
