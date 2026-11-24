pub mod ast;
pub mod borrow;
pub mod build_sys;
pub mod cfg;
pub mod cfg_analysis;
pub mod check;
pub mod codegen_gpu;
pub mod codegen_x64;
pub mod comptime;
pub mod diag;
pub mod effects;
pub mod eir;
pub mod eir_interp;
pub mod elf64;
pub mod emi;
pub mod generics;
pub mod lexer;
pub mod parser;
pub mod resolve;
pub mod span;
pub mod tcb_audit;
pub mod types;
pub mod ui_test;
pub mod verify;
pub mod x64_asm;

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use ast::CompilationUnit;
use cfg::{lower_program, CfgFnKey, CfgProgram, CfgTerm};
use cfg_analysis::analyze_program;
use check::{check_program, TypeTable};
pub use codegen_x64::X64Options;
use diag::DiagBag;
pub use effects::EffectsReport;
use lexer::{lex_file, TokenKind};
use parser::parse_file;
use resolve::{resolve_program, ResolvedProgram};
use span::{Interner, SourceMap};
pub use types::dump_types_canonical;
use types::{build_type_env, GenericVal, TypeEnv};
pub use verify::VerifyReport;

pub struct ParseResult {
    pub unit: CompilationUnit,
    pub lines: usize,
}

pub struct ResolveResult {
    pub units: Vec<CompilationUnit>,
    pub resolved: ResolvedProgram,
    pub total_lines: usize,
}

pub fn resolve_files_in_mem(
    files: &[(String, Vec<u8>)],
    sm: &mut SourceMap,
    interner: &mut Interner,
    diags: &mut DiagBag,
) -> ResolveResult {
    let mut units = Vec::with_capacity(files.len());
    let mut profiles = Vec::with_capacity(files.len());
    let mut total_lines = 0usize;

    for (path, bytes) in files {
        let fid = sm.add_file_bytes(path.clone(), bytes.clone());
        let file = sm.file(fid);
        total_lines += file.line_count() as usize;
        profiles.push(file.profile);
        let lexed = lex_file(file, diags);
        let unit = parse_file(file, &lexed, interner, diags);
        units.push(unit);
    }

    let resolved = resolve_program(&units, &profiles, interner, diags);
    ResolveResult {
        units,
        resolved,
        total_lines,
    }
}

pub struct CheckResult {
    pub units: Vec<CompilationUnit>,
    pub resolved: ResolvedProgram,
    pub env: TypeEnv,
    pub type_table: TypeTable,
    pub cfg: CfgProgram,
    pub eir: eir::EirProgram,
    pub effects: EffectsReport,
    pub verify: VerifyReport,
    pub total_lines: usize,
}

pub fn check_files_in_mem(
    files: &[(String, Vec<u8>)],
    sm: &mut SourceMap,
    interner: &mut Interner,
    diags: &mut DiagBag,
) -> CheckResult {
    let mut units = Vec::with_capacity(files.len());
    let mut profiles = Vec::with_capacity(files.len());
    let mut total_lines = 0usize;

    for (path, bytes) in files {
        let fid = sm.add_file_bytes(path.clone(), bytes.clone());
        let file = sm.file(fid);
        total_lines += file.line_count() as usize;
        profiles.push(file.profile);
        let lexed = lex_file(file, diags);
        let unit = parse_file(file, &lexed, interner, diags);
        units.push(unit);
    }

    let mut resolved = resolve_program(&units, &profiles, interner, diags);
    let mut env = build_type_env(&units, &mut resolved, interner, diags);
    let type_table = check_program(&units, &mut resolved, &mut env, interner, diags);
    let pre_err_count = diags.diagnostics().len();
    let cfg = lower_program(&units, &resolved, &mut env, &type_table, interner);
    analyze_program(&units, &resolved, &env, &type_table, &cfg, interner, diags);
    let effects = if pre_err_count == 0 {
        effects::check_effects_and_concurrency(&units, &resolved, &env, &cfg, interner, diags)
    } else {
        EffectsReport::default()
    };
    let mut eir = eir::lower_program_for_verify(&cfg, &resolved, &env, diags);
    let verify = if pre_err_count == 0 {
        let vr = verify::verify_program(&mut eir, &env, interner, diags);
        generics::apply_layout_sharing(&mut eir, &env);
        for f in &eir.functions {
            if f.sig.effects.is_gpu {
                codegen_gpu::validate_gpu_function(f, &env, interner, diags);
            }
        }
        vr
    } else {
        VerifyReport::default()
    };

    CheckResult {
        units,
        resolved,
        env,
        type_table,
        cfg,
        eir,
        effects,
        verify,
        total_lines,
    }
}

pub fn dump_cfg_canonical(cfg: &CfgProgram, env: &TypeEnv, interner: &Interner) -> String {
    let mut entries: Vec<(String, String)> = Vec::with_capacity(cfg.functions.len());

    for f in &cfg.functions {
        let kind_str = match f.key {
            CfgFnKey::TopDef(_) => {
                if f.is_sys {
                    "sys"
                } else {
                    "fn"
                }
            }
            CfgFnKey::ImplMethod { .. } => "method",
            CfgFnKey::GenericInst(_) => "inst",
            CfgFnKey::Closure(_) => "closure",
        };
        let name = interner.get(f.name);
        let gstr = match &f.base_generic {
            Some((_, args)) if !args.is_empty() => {
                let parts: Vec<String> = args
                    .iter()
                    .map(|a| match a {
                        GenericVal::Ty(t) => env.format_ty(*t, interner),
                        GenericVal::Int(n) => n.to_string(),
                        GenericVal::Variant(s) => interner.get(*s).to_owned(),
                    })
                    .collect();
                format!("[{}]", parts.join(", "))
            }
            _ => String::new(),
        };
        let param_tys: Vec<String> = f
            .sig
            .params
            .iter()
            .map(|p| env.format_ty(p.ty, interner))
            .collect();
        let ret_ty = env.format_ty(f.sig.ret, interner);

        let mut buf = String::new();
        let _ = writeln!(
            buf,
            "cfg {kind_str} {name}{gstr}({}) -> {ret_ty} locals={} blocks={}",
            param_tys.join(", "),
            f.locals.len(),
            f.blocks.len()
        );
        for b in &f.blocks {
            let _ = write!(buf, "  bb{}: stmts={} term=", b.id.0, b.stmts.len());
            match &b.term {
                CfgTerm::Goto(t) => {
                    let _ = writeln!(buf, "goto bb{}", t.0);
                }
                CfgTerm::Branch { then_bb, else_bb, .. } => {
                    let _ = writeln!(buf, "branch bb{} bb{}", then_bb.0, else_bb.0);
                }
                CfgTerm::SwitchInt { targets, otherwise, .. } => {
                    buf.push_str("switch [");
                    for (i, &(val, bb)) in targets.iter().enumerate() {
                        if i > 0 {
                            buf.push_str(", ");
                        }
                        let _ = write!(buf, "{val}->bb{}", bb.0);
                    }
                    let _ = writeln!(buf, "] else bb{}", otherwise.0);
                }
                CfgTerm::Return { value: Some(_), .. } => buf.push_str("return val\n"),
                CfgTerm::Return { value: None, .. } => buf.push_str("return void\n"),
                CfgTerm::Unreachable { explicit: true, .. } => {
                    buf.push_str("unreachable(explicit)\n");
                }
                CfgTerm::Unreachable { explicit: false, .. } => buf.push_str("unreachable\n"),
            }
        }
        let sort_key = format!(
            "{name}\t{kind_str}\t{gstr}\t({})\t{ret_ty}\t{}\t{}",
            param_tys.join(", "),
            f.locals.len(),
            f.blocks.len()
        );
        entries.push((sort_key, buf));
    }

    entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut out = String::new();
    for (_, text) in entries {
        out.push_str(&text);
    }
    out
}

pub fn dump_verify_canonical(
    cfg: &CfgProgram,
    effects: &EffectsReport,
    verify: &VerifyReport,
    interner: &Interner,
) -> String {
    let mut entries: Vec<(String, String)> = Vec::with_capacity(cfg.functions.len());

    for f in &cfg.functions {
        let kind_str = match f.key {
            CfgFnKey::TopDef(_) => {
                if f.is_sys {
                    "sys"
                } else {
                    "fn"
                }
            }
            CfgFnKey::ImplMethod { .. } => "method",
            CfgFnKey::GenericInst(_) => "inst",
            CfgFnKey::Closure(_) => "closure",
        };
        let name = interner.get(f.name);
        let eff = effects
            .functions
            .get(&f.key)
            .or_else(|| {
                f.base_generic
                    .as_ref()
                    .and_then(|(bk, _)| effects.functions.get(bk))
            })
            .cloned()
            .unwrap_or_default();
        let ver = verify.functions.get(&f.key).cloned().unwrap_or_default();

        let mut eff_tags: Vec<&'static str> = Vec::new();
        if eff.can_mutate {
            eff_tags.push("mut");
        }
        if eff.can_alloc {
            eff_tags.push("alloc");
        }
        if eff.can_block {
            eff_tags.push("block");
        }
        if eff.uses_extern_c {
            eff_tags.push("extern_c");
        }
        if eff_tags.is_empty() {
            eff_tags.push("pure");
        }
        let eff_str = eff_tags.join(",");

        let line = format!(
            "verify {kind_str} {name} effects=[{eff_str}] is_recursive={} stack_bytes={} checks={}/{}\n",
            eff.is_recursive, eff.stack_bytes, ver.proven_checks, ver.total_checks
        );
        let sort_key = format!(
            "{name}\t{kind_str}\t{}\t{eff_str}\t{}\t{}\t{}\t{}",
            f.params.len(),
            eff.is_recursive,
            eff.stack_bytes,
            ver.proven_checks,
            ver.total_checks
        );
        entries.push((sort_key, line));
    }

    entries.sort_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(&b.1)));
    let mut out = String::new();
    for (_, line) in entries {
        out.push_str(&line);
    }
    out
}

pub struct EmitNativeResult {
    pub check: CheckResult,
    pub eir: eir::EirProgram,
    pub elf: elf64::LinkedElf,
    pub stats: Option<emi::IncrementalStats>,
}

pub fn compile_files_to_x64_elf(
    files: &[(String, Vec<u8>)],
    sm: &mut SourceMap,
    interner: &mut Interner,
    diags: &mut DiagBag,
    opts: &codegen_x64::X64Options,
    emi_cache_dir: Option<&Path>,
) -> Result<EmitNativeResult, String> {
    let check = check_files_in_mem(files, sm, interner, diags);
    if diags.has_errors() {
        return Err(diags.render_all(sm));
    }
    for f in &check.cfg.functions {
        if !f.sig.generics.is_empty() || f.sig.is_comptime {
            continue;
        }
        for u in &f.unsupported {
            diags.error(
                diag::ErrorCode::E0230,
                u.span,
                format!("cannot lower `{}` to EIR/x86_64", u.kind),
            );
        }
    }
    if diags.has_errors() {
        return Err(diags.render_all(sm));
    }
    let eir = check.eir.clone();
    eir::lint_eir(&eir, &check.env)?;
    let (elf, stats) = if let Some(cdir) = emi_cache_dir {
        let (linked, st) = emi::compile_x64_incremental(
            &eir,
            &check.resolved,
            &check.env,
            sm,
            interner,
            opts,
            cdir,
        )?;
        (linked, Some(st))
    } else {
        let linked = codegen_x64::compile_and_link_x64(
            &eir,
            &check.resolved,
            &check.env,
            sm,
            interner,
            opts,
        )?;
        (linked, None)
    };
    Ok(EmitNativeResult {
        check,
        eir,
        elf,
        stats,
    })
}

/// Writes an ELF64 artifact to `out_path`, setting `0o755` on Unix for executables or `0o644` for shared libraries.
pub fn write_elf_file(bytes: &[u8], out_path: &Path, executable: bool) -> Result<(), String> {
    if let Some(parent) = out_path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create dir `{}`: {e}", parent.display()))?;
    }
    fs::write(out_path, bytes)
        .map_err(|e| format!("failed to write `{}`: {e}", out_path.display()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let mut perms = fs::metadata(out_path)
            .map_err(|e| format!("failed to stat `{}`: {e}", out_path.display()))?
            .permissions();
        perms.set_mode(if executable { 0o755 } else { 0o644 });
        fs::set_permissions(out_path, perms)
            .map_err(|e| format!("failed to chmod `{}`: {e}", out_path.display()))?;
    }
    #[cfg(not(unix))]
    let _ = executable;
    Ok(())
}

pub fn write_executable_elf(bytes: &[u8], out_bin: &Path) -> Result<(), String> {
    write_elf_file(bytes, out_bin, true)
}

pub fn write_shared_elf(bytes: &[u8], out_so: &Path) -> Result<(), String> {
    write_elf_file(bytes, out_so, false)
}

pub fn interpret_files(
    files: &[(String, Vec<u8>)],
    sm: &mut SourceMap,
    interner: &mut Interner,
    diags: &mut DiagBag,
) -> Result<eir_interp::InterpResult, String> {
    interpret_files_with_args(files, sm, interner, diags, &[])
}

pub fn interpret_files_with_args(
    files: &[(String, Vec<u8>)],
    sm: &mut SourceMap,
    interner: &mut Interner,
    diags: &mut DiagBag,
    prog_args: &[String],
) -> Result<eir_interp::InterpResult, String> {
    let check = check_files_in_mem(files, sm, interner, diags);
    if diags.has_errors() {
        return Err(diags.render_all(sm));
    }
    for f in &check.cfg.functions {
        if !f.sig.generics.is_empty() || f.sig.is_comptime {
            continue;
        }
        for u in &f.unsupported {
            diags.error(
                diag::ErrorCode::E0230,
                u.span,
                format!("cannot lower `{}` to EIR", u.kind),
            );
        }
    }
    if diags.has_errors() {
        return Err(diags.render_all(sm));
    }
    eir::lint_eir(&check.eir, &check.env)?;
    eir_interp::interpret_eir_with_args(
        &check.eir,
        &check.resolved,
        &check.env,
        interner,
        eir_interp::DEFAULT_STEP_LIMIT,
        prog_args,
    )
}

pub fn compile_files_to_spirv(
    files: &[(String, Vec<u8>)],
    sm: &mut SourceMap,
    interner: &mut Interner,
    diags: &mut DiagBag,
) -> Result<Vec<codegen_gpu::SpirvModule>, String> {
    let check = check_files_in_mem(files, sm, interner, diags);
    if diags.has_errors() {
        return Err(diags.render_all(sm));
    }
    let spvs = codegen_gpu::compile_gpu_shaders(&check.eir, &check.env, interner, diags);
    if diags.has_errors() {
        return Err(diags.render_all(sm));
    }
    Ok(spvs)
}

pub fn run_compiled_binary(bin: &Path) -> Result<(i32, String, String), String> {
    run_compiled_binary_with_args(bin, &[])
}

pub fn run_compiled_binary_with_args(
    bin: &Path,
    args: &[String],
) -> Result<(i32, String, String), String> {
    let mut last_err = None;
    for attempt in 0..8 {
        match Command::new(bin).args(args).output() {
            Ok(out) => {
                let code = out.status.code().unwrap_or(-1);
                let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
                let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
                return Ok((code, stdout, stderr));
            }
            Err(e) => {
                // Retry briefly on Linux `ETXTBSY` (errno 26) caused by concurrent test fork() inheriting write FDs.
                if e.raw_os_error() == Some(26) && attempt + 1 < 8 {
                    std::thread::sleep(std::time::Duration::from_millis(10));
                    last_err = Some(e);
                    continue;
                }
                return Err(format!("failed to run `{}`: {e}", bin.display()));
            }
        }
    }
    Err(format!(
        "failed to run `{}`: {}",
        bin.display(),
        last_err.map_or_else(|| "unknown error".into(), |e| e.to_string())
    ))
}

/// Parses `files` and produces a [`tcb_audit::TcbAuditReport`] over all `@hal` / `.hal.eon` units.
pub fn audit_tcb_files(files: &[(String, Vec<u8>)]) -> Result<tcb_audit::TcbAuditReport, String> {
    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let mut diags = DiagBag::new();
    let mut units = Vec::with_capacity(files.len());
    for (path, bytes) in files {
        let parsed = parse_bytes(path, bytes.clone(), &mut sm, &mut interner, &mut diags);
        units.push(parsed.unit);
    }
    if diags.has_errors() {
        return Err(diags.render_all(&sm));
    }
    Ok(tcb_audit::audit_tcb(&units, &sm, &interner))
}

pub fn load_eon_sources(paths: &[PathBuf]) -> Result<Vec<(String, Vec<u8>)>, String> {
    load_eon_sources_with_opts(paths, true)
}

pub fn load_eon_sources_no_stdlib(paths: &[PathBuf]) -> Result<Vec<(String, Vec<u8>)>, String> {
    load_eon_sources_with_opts(paths, false)
}

fn load_eon_sources_with_opts(
    paths: &[PathBuf],
    auto_stdlib: bool,
) -> Result<Vec<(String, Vec<u8>)>, String> {
    let mut files = Vec::new();
    for pb in paths {
        collect_eon_files(pb, &mut files)
            .map_err(|e| format!("failed to collect `{}`: {e}", pb.display()))?;
    }
    let mut loaded = Vec::with_capacity(files.len());
    let mut canonical_seen = BTreeSet::new();
    for f in &files {
        let canon = fs::canonicalize(f).unwrap_or_else(|_| f.clone());
        canonical_seen.insert(canon);
        let bytes = fs::read(f).map_err(|e| format!("failed to read `{}`: {e}", f.display()))?;
        loaded.push((f.to_string_lossy().into_owned(), bytes));
    }

    if !auto_stdlib {
        return Ok(loaded);
    }

    // Transitively auto-discover `core.*` and `std.*` modules from the workspace stdlib root.
    let stdlib_root = find_stdlib_root(paths);
    if let Some(root) = stdlib_root {
        let mut seen_mods: BTreeSet<(String, Vec<String>)> = BTreeSet::new();
        let mut scan_idx = 0usize;
        while scan_idx < loaded.len() {
            let imports = {
                let (path, bytes) = &loaded[scan_idx];
                collect_stdlib_imports(path, bytes)
            };
            scan_idx += 1;
            for (pkg, segs) in imports {
                if !seen_mods.insert((pkg.clone(), segs.clone())) {
                    continue;
                }
                if let Some(p) = resolve_stdlib_module_file(&root, &pkg, &segs) {
                    let canon = fs::canonicalize(&p).unwrap_or_else(|_| p.clone());
                    if canonical_seen.insert(canon) {
                        let bytes = fs::read(&p)
                            .map_err(|e| format!("failed to read `{}`: {e}", p.display()))?;
                        loaded.push((p.to_string_lossy().into_owned(), bytes));
                    }
                }
            }
        }
    }
    Ok(loaded)
}

fn collect_stdlib_imports(path: &str, bytes: &[u8]) -> Vec<(String, Vec<String>)> {
    let mut sm = SourceMap::new();
    let fid = sm.add_file_bytes(path, bytes.to_vec());
    let sf = sm.file(fid);
    let mut diags = DiagBag::new();
    let lexed = lex_file(sf, &mut diags);
    let text = sf.text();
    let tokens = &lexed.tokens;
    let mut out = Vec::new();

    let mut i = 0usize;
    while i < tokens.len() {
        if tokens[i].kind != TokenKind::KwImport {
            i += 1;
            continue;
        }
        i += 1;
        if i >= tokens.len() || tokens[i].kind != TokenKind::Ident {
            continue;
        }
        let root_pkg = &text[tokens[i].span.start as usize..tokens[i].span.end as usize];
        if root_pkg != "core" && root_pkg != "std" {
            continue;
        }
        i += 1;
        if i >= tokens.len() || tokens[i].kind != TokenKind::Dot {
            continue;
        }
        i += 1;
        if i >= tokens.len() {
            break;
        }
        if tokens[i].kind == TokenKind::LBrace {
            i += 1;
            let mut expect_item = true;
            while i < tokens.len() {
                match tokens[i].kind {
                    TokenKind::RBrace | TokenKind::Semi | TokenKind::Eof => {
                        i += 1;
                        break;
                    }
                    TokenKind::Comma => {
                        expect_item = true;
                        i += 1;
                    }
                    TokenKind::KwAs => {
                        expect_item = false;
                        i += 1;
                        if i < tokens.len() && tokens[i].kind == TokenKind::Ident {
                            i += 1;
                        }
                    }
                    TokenKind::Ident if expect_item => {
                        let sub = &text[tokens[i].span.start as usize..tokens[i].span.end as usize];
                        out.push((root_pkg.to_owned(), vec![sub.to_owned()]));
                        expect_item = false;
                        i += 1;
                    }
                    _ => {
                        i += 1;
                    }
                }
            }
        } else if tokens[i].kind == TokenKind::Ident {
            let mut segs = Vec::new();
            let first = &text[tokens[i].span.start as usize..tokens[i].span.end as usize];
            segs.push(first.to_owned());
            i += 1;
            while i + 1 < tokens.len()
                && tokens[i].kind == TokenKind::Dot
                && tokens[i + 1].kind == TokenKind::Ident
            {
                let seg = &text[tokens[i + 1].span.start as usize..tokens[i + 1].span.end as usize];
                segs.push(seg.to_owned());
                i += 2;
            }
            out.push((root_pkg.to_owned(), segs));
        }
    }

    out
}

fn resolve_stdlib_module_file(root: &Path, pkg: &str, segs: &[String]) -> Option<PathBuf> {
    if segs.is_empty() {
        return None;
    }
    let mut dir = root.join(pkg);
    for (idx, seg) in segs.iter().enumerate() {
        if idx + 1 == segs.len() {
            let cand_eon = dir.join(format!("{seg}.eon"));
            if cand_eon.is_file() {
                return Some(cand_eon);
            }
            let cand_hal = dir.join(format!("{seg}.hal.eon"));
            if cand_hal.is_file() {
                return Some(cand_hal);
            }
        } else {
            dir = dir.join(seg);
        }
    }
    None
}

fn dir_has_eon_file(dir: &Path) -> bool {
    let Ok(entries) = fs::read_dir(dir) else {
        return false;
    };
    for entry in entries.flatten() {
        let p = entry.path();
        if p.is_file()
            && let Some(name) = p.file_name().and_then(|s| s.to_str())
            && name.ends_with(".eon")
        {
            return true;
        }
    }
    false
}

fn is_stdlib_root(dir: &Path) -> bool {
    dir_has_eon_file(&dir.join("core")) || dir_has_eon_file(&dir.join("std"))
}

fn find_stdlib_root_from(start: &Path) -> Option<PathBuf> {
    let abs = fs::canonicalize(start).unwrap_or_else(|_| start.to_path_buf());
    let mut cur = if abs.is_file() {
        abs.parent().map(Path::to_path_buf)
    } else {
        Some(abs)
    };
    while let Some(dir) = cur {
        if is_stdlib_root(&dir) {
            return Some(dir);
        }
        cur = dir.parent().map(Path::to_path_buf);
    }
    None
}

fn find_stdlib_root(paths: &[PathBuf]) -> Option<PathBuf> {
    for pb in paths {
        if let Some(root) = find_stdlib_root_from(pb) {
            return Some(root);
        }
    }
    if let Ok(cwd) = std::env::current_dir()
        && let Some(root) = find_stdlib_root_from(&cwd)
    {
        return Some(root);
    }
    if let Ok(exe) = std::env::current_exe()
        && let Some(root) = find_stdlib_root_from(&exe)
    {
        return Some(root);
    }
    None
}

pub fn parse_bytes(
    path: &str,
    bytes: Vec<u8>,
    sm: &mut SourceMap,
    interner: &mut Interner,
    diags: &mut DiagBag,
) -> ParseResult {
    let file_id = sm.add_file_bytes(path, bytes);
    let file = sm.file(file_id);
    let lines = file.line_count() as usize;
    let lexed = lex_file(file, diags);
    let unit = parse_file(file, &lexed, interner, diags);
    ParseResult { unit, lines }
}

pub fn collect_eon_files(path: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    if path.is_file() {
        out.push(path.to_path_buf());
        return Ok(());
    }
    let mut entries: Vec<_> = fs::read_dir(path)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|e| e.path());
    for entry in entries {
        let p = entry.path();
        if p.is_dir() {
            collect_eon_files(&p, out)?;
        } else if let Some(name) = p.file_name().and_then(|s| s.to_str())
            && name.ends_with(".eon")
        {
            out.push(p);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unique_tmp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("eon_lib_tests_{}_{}", tag, std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        fs::create_dir_all(&dir).expect("create test tmp dir");
        dir
    }

    #[test]
    fn load_eon_sources_discovers_transitive_core_and_std_modules() {
        let root = unique_tmp_dir("transitive_stdlib");
        fs::create_dir_all(root.join("core")).expect("mkdir core");
        fs::create_dir_all(root.join("std")).expect("mkdir std");
        fs::create_dir_all(root.join("app")).expect("mkdir app");

        fs::write(
            root.join("core/sync.hal.eon"),
            "module core.sync @hal;\npub fn fence() {}\n",
        )
        .expect("write core/sync");
        fs::write(
            root.join("core/io.eon"),
            "module core.io;\nimport core.sync;\npub fn ping() {}\n",
        )
        .expect("write core/io");
        fs::write(
            root.join("core/str.eon"),
            "module core.str;\npub fn len_ok() -> bool { return true; }\n",
        )
        .expect("write core/str");
        fs::write(
            root.join("core/hal.hal.eon"),
            "module core.hal @hal;\npub fn noop() {}\n",
        )
        .expect("write core/hal");
        fs::write(
            root.join("std/sched.hal.eon"),
            "module std.sched @hal;\nimport core.io;\n",
        )
        .expect("write std/sched");
        fs::write(
            root.join("std/rt.hal.eon"),
            "module std.rt @hal;\nimport core.str;\n",
        )
        .expect("write std/rt");
        fs::write(
            root.join("app/main.eon"),
            "// import core.hal;\n\
             /* import core.nonexistent; */\n\
             module app.main;\n\
             import std.{sched, rt as runtime};\n\
             pub fn main() -> i32 { return 0_i32; }\n",
        )
        .expect("write app/main");

        let loaded = load_eon_sources(&[root.join("app/main.eon")]).expect("load sources");
        let names: Vec<&str> = loaded.iter().map(|(p, _)| p.as_str()).collect();
        assert!(names.iter().any(|p| p.ends_with("app/main.eon")));
        assert!(names.iter().any(|p| p.ends_with("std/sched.hal.eon")));
        assert!(names.iter().any(|p| p.ends_with("std/rt.hal.eon")));
        assert!(names.iter().any(|p| p.ends_with("core/io.eon")));
        assert!(names.iter().any(|p| p.ends_with("core/str.eon")));
        assert!(names.iter().any(|p| p.ends_with("core/sync.hal.eon")));
        assert!(
            !names.iter().any(|p| p.ends_with("core/hal.hal.eon")),
            "commented import core.hal must be ignored"
        );

        let _ = fs::remove_dir_all(&root);
    }

    #[test]
    fn audit_tcb_files_parses_and_reports_hal_units() {
        let files = vec![
            (
                "app/main.eon".to_owned(),
                b"module app.main;\npub fn main() -> i32 { return 0_i32; }\n".to_vec(),
            ),
            (
                "drv/io.hal.eon".to_owned(),
                b"module drv.io @hal;\npub fn tick() { raw { asm(\"nop\"); } }\n".to_vec(),
            ),
        ];
        let report = audit_tcb_files(&files).expect("audit_tcb_files");
        assert_eq!(report.files.len(), 1);
        assert_eq!(report.files[0].module_name, "drv.io");
        assert_eq!(report.total_raw_lines, 1);
        assert_eq!(report.total_asm_sites, 1);
    }
}
