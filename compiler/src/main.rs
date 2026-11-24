use std::collections::BTreeSet;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Instant;

use eonc::{
    audit_tcb_files, build_sys, collect_eon_files, compile_files_to_spirv,
    compile_files_to_x64_elf, dump_cfg_canonical, dump_types_canonical, dump_verify_canonical,
    interpret_files_with_args, load_eon_sources, load_eon_sources_no_stdlib, resolve_files_in_mem,
    run_compiled_binary_with_args, tcb_audit, write_executable_elf, write_shared_elf,
    EmitNativeResult, X64Options,
};
use eonc::build_sys::BuildTargetKind;
use eonc::diag::DiagBag;
use eonc::elf64::ElfMode;
use eonc::lexer::lex_file;
use eonc::parser::parse_file;
use eonc::span::{Interner, SourceMap};
use eonc::ui_test::run_ui_dir;

fn print_usage() {
    eprintln!(
        "eonc — Eon compiler (v0.2)\n\
         \n\
         USAGE:\n\
         \teonc lex [--dump] <path...>\n\
         \teonc parse [--dump-ast | --dump-ast-full] [--repeat N] <path...>\n\
         \teonc resolve [--dump] [--no-stdlib] <path...>\n\
         \teonc check [--dump-types] [--dump-cfg] [--dump-eir] [--dump-verify] [--no-stdlib] <path...>\n\
         \teonc emit-spirv [-o out.spv] <path...>\n\
         \teonc build [--native] [--freestanding] [--shared] [--soname NAME] [-l LIB] [--sanitize] [--audit-tcb] [--emi-cache DIR] [-o out_path] [path...|build.eon]\n\
         \teonc run [--native | --interp] [--sanitize] [-l LIB] [--emi-cache DIR] <path...> [-- <prog_args...>]\n\
         \teonc test [tests/ui]\n"
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        print_usage();
        return ExitCode::FAILURE;
    }

    match args[0].as_str() {
        "lex" => cmd_lex(&args[1..]),
        "parse" => cmd_parse(&args[1..]),
        "resolve" => cmd_resolve(&args[1..]),
        "check" => cmd_check(&args[1..]),
        "emit-spirv" | "--emit-spirv" => cmd_emit_spirv(&args[1..]),
        "build" => cmd_build(&args[1..]),
        "run" => cmd_run(&args[1..]),
        "test" => cmd_test(&args[1..]),
        "-h" | "--help" | "help" => {
            print_usage();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("unknown subcommand `{}`", other);
            print_usage();
            ExitCode::FAILURE
        }
    }
}

fn cmd_lex(args: &[String]) -> ExitCode {
    if args.is_empty() {
        eprintln!("usage: eonc lex [--dump] <path...>");
        return ExitCode::FAILURE;
    }

    if args.iter().any(|a| a == "--dump") {
        let mut inputs = Vec::new();
        for arg in args {
            match arg.as_str() {
                "--dump" => {}
                "-h" | "--help" => {
                    eprintln!("usage: eonc lex [--dump] <path...>");
                    return ExitCode::SUCCESS;
                }
                other if other.starts_with('-') => {
                    eprintln!("unknown lex flag `{other}`");
                    return ExitCode::FAILURE;
                }
                path => inputs.push(PathBuf::from(path)),
            }
        }
        if inputs.is_empty() {
            eprintln!("usage: eonc lex --dump <path...>");
            return ExitCode::FAILURE;
        }
        let mut files = Vec::new();
        for p in &inputs {
            if let Err(e) = collect_eon_files(p, &mut files) {
                eprintln!("failed to collect `{}`: {}", p.display(), e);
                return ExitCode::FAILURE;
            }
        }
        files.sort();

        let mut had_errors = false;
        for f in &files {
            let name = f.to_string_lossy().to_string();
            let bytes = match fs::read(f) {
                Ok(b) => b,
                Err(e) => {
                    eprintln!("failed to read `{}`: {}", f.display(), e);
                    return ExitCode::FAILURE;
                }
            };
            println!("=== {} ===", name);
            let mut sm = SourceMap::new();
            let fid = sm.add_file_bytes(name, bytes);
            let file = sm.file(fid);
            let mut diags = DiagBag::new();
            let lexed = lex_file(file, &mut diags);
            for tok in &lexed.tokens {
                let (line, col) = file.line_col(tok.span.start);
                let text = &file.text()[tok.span.start as usize..tok.span.end as usize];
                println!("{}:{}\t{:?}\t{:?}", line, col, tok.kind, text);
            }
            if diags.has_errors() {
                eprint!("{}", diags.render_all(&sm));
                had_errors = true;
            }
        }
        return if had_errors {
            ExitCode::FAILURE
        } else {
            ExitCode::SUCCESS
        };
    }

    let path = &args[0];
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("failed to read `{}`: {}", path, e);
            return ExitCode::FAILURE;
        }
    };
    let mut sm = SourceMap::new();
    let fid = sm.add_file_bytes(path, bytes);
    let file = sm.file(fid);
    let mut diags = DiagBag::new();
    let lexed = lex_file(file, &mut diags);

    for tok in &lexed.tokens {
        let (line, col) = file.line_col(tok.span.start);
        let text = &file.text()[tok.span.start as usize..tok.span.end as usize];
        println!("{}:{}\t{:?}\t{:?}", line, col, tok.kind, text);
    }

    if diags.has_errors() {
        eprint!("{}", diags.render_all(&sm));
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

fn cmd_parse(args: &[String]) -> ExitCode {
    let mut dump_ast = false;
    let mut dump_ast_full = false;
    let mut repeat: usize = 1;
    let mut inputs = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--dump-ast" => dump_ast = true,
            "--dump-ast-full" => dump_ast_full = true,
            "--repeat" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--repeat requires an integer argument");
                    return ExitCode::FAILURE;
                }
                repeat = match args[i].parse() {
                    Ok(n) if n >= 1 => n,
                    _ => {
                        eprintln!("invalid --repeat count `{}`", args[i]);
                        return ExitCode::FAILURE;
                    }
                };
            }
            "-h" | "--help" => {
                eprintln!("usage: eonc parse [--dump-ast | --dump-ast-full] [--repeat N] <path...>");
                return ExitCode::SUCCESS;
            }
            path => inputs.push(PathBuf::from(path)),
        }
        i += 1;
    }

    if inputs.is_empty() {
        eprintln!("usage: eonc parse [--dump-ast | --dump-ast-full] [--repeat N] <path...>");
        return ExitCode::FAILURE;
    }

    let mut files = Vec::new();
    for p in &inputs {
        if let Err(e) = collect_eon_files(p, &mut files) {
            eprintln!("failed to collect `{}`: {}", p.display(), e);
            return ExitCode::FAILURE;
        }
    }

    let mut loaded = Vec::with_capacity(files.len());
    for f in &files {
        match fs::read(f) {
            Ok(b) => loaded.push((f.to_string_lossy().to_string(), b)),
            Err(e) => {
                eprintln!("failed to read `{}`: {}", f.display(), e);
                return ExitCode::FAILURE;
            }
        }
    }

    let start = Instant::now();
    let mut total_lines = 0usize;
    let mut had_errors = false;

    for run_idx in 0..repeat {
        let mut sm = SourceMap::new();
        let mut interner = Interner::new();
        let mut diags = DiagBag::new();

        for (name, bytes) in &loaded {
            let fid = sm.add_file_bytes(name.clone(), bytes.clone());
            let file = sm.file(fid);
            total_lines += file.line_count() as usize;
            let lexed = lex_file(file, &mut diags);
            let unit = parse_file(file, &lexed, &mut interner, &mut diags);
            if dump_ast && run_idx == 0 {
                println!("=== {} ===\n{}", name, unit.dump(&interner));
            }
            if dump_ast_full && run_idx == 0 {
                print!("=== {} ===\n{}", name, unit.dump_full(&interner));
            }
        }

        if run_idx == 0 && diags.has_errors() {
            eprint!("{}", diags.render_all(&sm));
            had_errors = true;
        }
    }

    let elapsed = start.elapsed();
    let secs = elapsed.as_secs_f64();
    let lps = if secs > 0.0 {
        (total_lines as f64 / secs) as u64
    } else {
        0
    };
    eprintln!(
        "parsed {} file(s), {} total lines in {:.2} ms ({} lines/sec)",
        loaded.len(),
        total_lines,
        secs * 1000.0,
        lps
    );

    if had_errors {
        ExitCode::FAILURE
    } else {
        ExitCode::SUCCESS
    }
}

fn cmd_resolve(args: &[String]) -> ExitCode {
    let mut dump = false;
    let mut no_stdlib = false;
    let mut inputs = Vec::new();

    for arg in args {
        match arg.as_str() {
            "--dump" => dump = true,
            "--no-stdlib" => no_stdlib = true,
            "-h" | "--help" => {
                eprintln!("usage: eonc resolve [--dump] [--no-stdlib] <path...>");
                return ExitCode::SUCCESS;
            }
            other if other.starts_with('-') => {
                eprintln!("unknown resolve flag `{other}`");
                return ExitCode::FAILURE;
            }
            path => inputs.push(PathBuf::from(path)),
        }
    }

    if inputs.is_empty() {
        eprintln!("usage: eonc resolve [--dump] [--no-stdlib] <path...>");
        return ExitCode::FAILURE;
    }

    let loaded = if no_stdlib {
        load_eon_sources_no_stdlib(&inputs)
    } else {
        load_eon_sources(&inputs)
    };
    let loaded = match loaded {
        Ok(l) => l,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let start = Instant::now();
    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let mut diags = DiagBag::new();

    let res = resolve_files_in_mem(&loaded, &mut sm, &mut interner, &mut diags);
    let secs = start.elapsed().as_secs_f64();

    if dump {
        print!("{}", res.resolved.dump_canonical(&interner));
    }

    if diags.has_errors() {
        eprint!("{}", diags.render_all(&sm));
        return ExitCode::FAILURE;
    }

    eprintln!(
        "resolved {} file(s), {} module(s), {} lines in {:.2} ms",
        loaded.len(),
        res.resolved.modules.len(),
        res.total_lines,
        secs * 1000.0
    );
    ExitCode::SUCCESS
}

fn cmd_check(args: &[String]) -> ExitCode {
    let mut dump_types = false;
    let mut dump_cfg = false;
    let mut dump_eir = false;
    let mut dump_verify = false;
    let mut no_stdlib = false;
    let mut paths = Vec::new();

    for arg in args {
        match arg.as_str() {
            "--dump-types" => dump_types = true,
            "--dump-cfg" => dump_cfg = true,
            "--dump-eir" => dump_eir = true,
            "--dump-verify" => dump_verify = true,
            "--no-stdlib" => no_stdlib = true,
            "-h" | "--help" => {
                eprintln!(
                    "usage: eonc check [--dump-types] [--dump-cfg] [--dump-eir] [--dump-verify] [--no-stdlib] <path...>"
                );
                return ExitCode::SUCCESS;
            }
            other if other.starts_with('-') => {
                eprintln!("unknown check flag `{other}`");
                return ExitCode::FAILURE;
            }
            path => paths.push(PathBuf::from(path)),
        }
    }

    if paths.is_empty() {
        eprintln!(
            "usage: eonc check [--dump-types] [--dump-cfg] [--dump-eir] [--dump-verify] [--no-stdlib] <path...>"
        );
        return ExitCode::FAILURE;
    }

    // Same source discovery as `build`/`run` (including auto-loaded `core.*` modules), so `check`
    // never disagrees with what actually gets compiled.
    let loaded = if no_stdlib {
        load_eon_sources_no_stdlib(&paths)
    } else {
        load_eon_sources(&paths)
    };
    let loaded = match loaded {
        Ok(l) => l,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let start = Instant::now();
    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let mut diags = DiagBag::new();

    let res = eonc::check_files_in_mem(&loaded, &mut sm, &mut interner, &mut diags);
    let secs = start.elapsed().as_secs_f64();

    if dump_types {
        print!("{}", dump_types_canonical(&res.resolved, &res.env, &interner));
    }
    if dump_cfg {
        print!("{}", dump_cfg_canonical(&res.cfg, &res.env, &interner));
    }
    if dump_eir {
        print!("{}", res.eir.dump_canonical(&res.env, &interner));
    }
    if dump_verify {
        print!(
            "{}",
            dump_verify_canonical(&res.cfg, &res.effects, &res.verify, &interner)
        );
    }

    if diags.has_errors() {
        eprint!("{}", diags.render_all(&sm));
        return ExitCode::FAILURE;
    }

    eprintln!(
        "checked {} file(s), {} module(s), {} lines in {:.2} ms",
        loaded.len(),
        res.resolved.modules.len(),
        res.total_lines,
        secs * 1000.0
    );
    ExitCode::SUCCESS
}

fn cmd_test(args: &[String]) -> ExitCode {
    let dir = if let Some(p) = args.first() {
        PathBuf::from(p)
    } else {
        let local = Path::new("tests/ui");
        if local.exists() {
            local.to_path_buf()
        } else {
            PathBuf::from("../tests/ui")
        }
    };

    if !dir.exists() {
        eprintln!("test directory `{}` does not exist", dir.display());
        return ExitCode::FAILURE;
    }

    let start = Instant::now();
    let summary = match run_ui_dir(&dir) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error scanning `{}`: {}", dir.display(), e);
            return ExitCode::FAILURE;
        }
    };
    let ms = start.elapsed().as_secs_f64() * 1000.0;

    for fail in &summary.failures {
        eprintln!("FAIL {}\n{}", fail.path.display(), fail.reason);
    }

    println!(
        "ui-test: {}/{} passed in {:.2} ms",
        summary.passed, summary.total, ms
    );

    if summary.failures.is_empty() && summary.total > 0 {
        ExitCode::SUCCESS
    } else {
        ExitCode::FAILURE
    }
}

fn parse_out_and_inputs(args: &[String]) -> Result<(Option<PathBuf>, Vec<PathBuf>), String> {
    let mut out = None;
    let mut inputs = Vec::new();
    let mut i = 0;
    while i < args.len() {
        if args[i] == "-o" {
            i += 1;
            if i >= args.len() {
                return Err("-o requires a path".into());
            }
            out = Some(PathBuf::from(&args[i]));
        } else {
            inputs.push(PathBuf::from(&args[i]));
        }
        i += 1;
    }
    if inputs.is_empty() {
        return Err("missing input <path...>".into());
    }
    Ok((out, inputs))
}

fn cmd_emit_spirv(args: &[String]) -> ExitCode {
    let (out_opt, inputs) = match parse_out_and_inputs(args) {
        Ok(x) => x,
        Err(e) => {
            eprintln!("usage: eonc emit-spirv [-o out.spv] <path...> ({e})");
            return ExitCode::FAILURE;
        }
    };
    let loaded = match load_eon_sources(&inputs) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let mut diags = DiagBag::new();
    let mods = match compile_files_to_spirv(&loaded, &mut sm, &mut interner, &mut diags) {
        Ok(r) => r,
        Err(e) => {
            eprint!("{e}");
            return ExitCode::FAILURE;
        }
    };
    if mods.is_empty() {
        eprintln!("no `@gpu` shader entry points found");
        return ExitCode::FAILURE;
    }
    let out_path = out_opt.unwrap_or_else(|| PathBuf::from("shader.spv"));
    if let Err(e) = fs::write(&out_path, &mods[0].bytes) {
        eprintln!("failed to write `{}`: {e}", out_path.display());
        return ExitCode::FAILURE;
    }
    eprintln!(
        "emitted SPIR-V `{}` ({} bytes, entry `{}`)",
        out_path.display(),
        mods[0].bytes.len(),
        mods[0].entry_name
    );
    ExitCode::SUCCESS
}

fn resolve_lib_soname(raw: &str) -> String {
    match raw {
        "c" => "libc.so.6".to_owned(),
        "m" => "libm.so.6".to_owned(),
        "pthread" => "libpthread.so.6".to_owned(),
        "dl" => "libdl.so.6".to_owned(),
        _ if raw.ends_with(".so") || raw.contains(".so.") || raw.contains('/') => raw.to_owned(),
        _ => format!("lib{raw}.so"),
    }
}

fn push_unique_lib(libs: &mut Vec<String>, lib: String) {
    if !libs.iter().any(|existing| existing == &lib) {
        libs.push(lib);
    }
}

fn report_built_elf(out_path: &Path, is_shared: bool, emitted: &EmitNativeResult) {
    let kind = if is_shared { "shared" } else { "native" };
    if let Some(st) = emitted.stats {
        eprintln!(
            "built {kind} ELF64 {} ({} bytes; emi: {}/{} cached, {} recompiled)",
            out_path.display(),
            emitted.elf.bytes.len(),
            st.fns_cached,
            st.fns_total,
            st.fns_recompiled
        );
    } else {
        eprintln!(
            "built {kind} ELF64 {} ({} bytes)",
            out_path.display(),
            emitted.elf.bytes.len()
        );
    }
}

fn cmd_build(args: &[String]) -> ExitCode {
    let mut out_opt: Option<PathBuf> = None;
    let mut freestanding = false;
    let mut shared = false;
    let mut soname: Option<String> = None;
    let mut needed_libs: Vec<String> = Vec::new();
    let mut sanitize = false;
    let mut audit_tcb = false;
    let mut emi_cache: Option<PathBuf> = None;
    let mut inputs = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "-o" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("-o requires a path");
                    return ExitCode::FAILURE;
                }
                out_opt = Some(PathBuf::from(&args[i]));
            }
            "--native" => {}
            "--freestanding" => freestanding = true,
            "--shared" => shared = true,
            "--soname" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--soname requires a library soname");
                    return ExitCode::FAILURE;
                }
                soname = Some(args[i].clone());
            }
            "-l" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("-l requires a library name");
                    return ExitCode::FAILURE;
                }
                push_unique_lib(&mut needed_libs, resolve_lib_soname(&args[i]));
            }
            "--sanitize" => sanitize = true,
            "--audit-tcb" => audit_tcb = true,
            "--emi-cache" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--emi-cache requires a directory");
                    return ExitCode::FAILURE;
                }
                emi_cache = Some(PathBuf::from(&args[i]));
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: eonc build [--native] [--freestanding] [--shared] [--soname NAME] [-l LIB] [--sanitize] [--audit-tcb] [--emi-cache DIR] [-o out_path] [path...|build.eon]"
                );
                return ExitCode::SUCCESS;
            }
            other => {
                if let Some(lib) = other.strip_prefix("-l") {
                    if lib.is_empty() {
                        eprintln!("-l requires a library name");
                        return ExitCode::FAILURE;
                    }
                    push_unique_lib(&mut needed_libs, resolve_lib_soname(lib));
                } else if other.starts_with('-') {
                    eprintln!("unknown build flag `{other}`");
                    return ExitCode::FAILURE;
                } else {
                    inputs.push(PathBuf::from(other));
                }
            }
        }
        i += 1;
    }

    if freestanding && shared {
        eprintln!("cannot combine --freestanding and --shared");
        return ExitCode::FAILURE;
    }

    let build_eon_ctx: Option<(PathBuf, PathBuf)> = if inputs.is_empty() {
        let candidate = Path::new("build.eon");
        if candidate.is_file() {
            Some((PathBuf::from("."), candidate.to_path_buf()))
        } else {
            None
        }
    } else if inputs.len() == 1
        && inputs[0].file_name().and_then(|s| s.to_str()) == Some("build.eon")
    {
        let bf = inputs[0].clone();
        let root = bf
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));
        Some((root, bf))
    } else {
        None
    };

    if let Some((pkg_root, build_file)) = build_eon_ctx {
        return run_build_eon(
            &pkg_root,
            &build_file,
            out_opt,
            freestanding,
            soname,
            needed_libs,
            sanitize,
            audit_tcb,
            emi_cache.as_deref(),
        );
    }

    if inputs.is_empty() {
        eprintln!(
            "usage: eonc build [--native] [--freestanding] [--shared] [--soname NAME] [-l LIB] [--sanitize] [--audit-tcb] [--emi-cache DIR] [-o out_path] [path...|build.eon]"
        );
        return ExitCode::FAILURE;
    }
    if soname.is_some() && !shared {
        eprintln!("--soname requires --shared");
        return ExitCode::FAILURE;
    }

    let mode = if freestanding {
        ElfMode::Freestanding {
            load_addr: 0x0010_0000,
            multiboot2: true,
        }
    } else if shared {
        ElfMode::LinuxShared {
            soname: soname.clone(),
        }
    } else {
        ElfMode::LinuxPie
    };
    let out_path = out_opt.clone().unwrap_or_else(|| {
        if shared {
            soname
                .as_ref()
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("libout.so"))
        } else {
            PathBuf::from("a.out")
        }
    });

    let loaded = match load_eon_sources(&inputs) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let opts = X64Options {
        mode,
        sanitize,
        needed_libs,
    };
    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let mut diags = DiagBag::new();
    let emitted = match compile_files_to_x64_elf(
        &loaded,
        &mut sm,
        &mut interner,
        &mut diags,
        &opts,
        emi_cache.as_deref(),
    ) {
        Ok(r) => r,
        Err(e) => {
            if audit_tcb
                && out_opt.is_none()
                && !shared
                && !freestanding
                && e.contains("unresolved entry symbol")
            {
                match audit_tcb_files(&loaded) {
                    Ok(report) => {
                        print!("{}", report.format_report());
                        return ExitCode::SUCCESS;
                    }
                    Err(audit_err) => {
                        eprint!("{audit_err}");
                        return ExitCode::FAILURE;
                    }
                }
            }
            eprint!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let write_res = if shared {
        write_shared_elf(&emitted.elf.bytes, &out_path)
    } else {
        write_executable_elf(&emitted.elf.bytes, &out_path)
    };
    if let Err(e) = write_res {
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }
    report_built_elf(&out_path, shared, &emitted);

    if audit_tcb {
        let report = tcb_audit::audit_tcb(&emitted.check.units, &sm, &interner);
        print!("{}", report.format_report());
    }
    ExitCode::SUCCESS
}

#[allow(clippy::too_many_arguments)]
fn run_build_eon(
    pkg_root: &Path,
    build_file: &Path,
    out_opt: Option<PathBuf>,
    freestanding: bool,
    soname: Option<String>,
    cli_needed_libs: Vec<String>,
    cli_sanitize: bool,
    cli_audit_tcb: bool,
    emi_cache: Option<&Path>,
) -> ExitCode {
    let plan = match build_sys::execute_build_eon(pkg_root, build_file) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    print!("{}", build_sys::format_build_summary(&plan));

    if let Some(ref o) = out_opt
        && plan.targets.len() > 1
        && !o.is_dir()
    {
        eprintln!(
            "-o `{}` cannot be a single file when build.eon declares {} targets",
            o.display(),
            plan.targets.len()
        );
        return ExitCode::FAILURE;
    }

    let out_dir_path = match build_sys::validate_sandbox_rel_path(pkg_root, &plan.out_dir) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };

    let sanitize = cli_sanitize || plan.sanitize;
    let audit_tcb = cli_audit_tcb || plan.audit_tcb;
    let mut needed_libs = Vec::new();
    for lib in &plan.needed_libs {
        push_unique_lib(&mut needed_libs, resolve_lib_soname(lib));
    }
    for lib in cli_needed_libs {
        push_unique_lib(&mut needed_libs, lib);
    }

    let mut combined_report = tcb_audit::TcbAuditReport::default();
    let mut seen_hal_files = BTreeSet::new();

    for target in &plan.targets {
        let target_root = match build_sys::validate_sandbox_rel_path(pkg_root, &target.root) {
            Ok(p) => p,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        };
        let loaded = match load_eon_sources(&[target_root]) {
            Ok(l) => l,
            Err(e) => {
                eprintln!("{e}");
                return ExitCode::FAILURE;
            }
        };

        let target_report = match audit_tcb_files(&loaded) {
            Ok(r) => r,
            Err(e) => {
                eprint!("{e}");
                return ExitCode::FAILURE;
            }
        };
        if let Err(e) = target_report.check_hal_whitelist(&plan.allow_hal, &["core", "std"]) {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
        for f in target_report.files {
            if seen_hal_files.insert((f.path.clone(), f.module_name.clone())) {
                combined_report.total_raw_lines += f.raw_lines;
                combined_report.total_asm_sites += f.asm_sites;
                combined_report.total_extern_c += f.extern_c_imports.len();
                combined_report.files.push(f);
            }
        }

        let mode = match target.kind {
            BuildTargetKind::Exe => {
                if freestanding {
                    ElfMode::Freestanding {
                        load_addr: 0x0010_0000,
                        multiboot2: true,
                    }
                } else {
                    ElfMode::LinuxPie
                }
            }
            BuildTargetKind::Shared => ElfMode::LinuxShared {
                soname: soname.clone().or_else(|| Some(target.name.clone())),
            },
        };
        let opts = X64Options {
            mode,
            sanitize,
            needed_libs: needed_libs.clone(),
        };

        let mut sm = SourceMap::new();
        let mut interner = Interner::new();
        let mut diags = DiagBag::new();
        let emitted = match compile_files_to_x64_elf(
            &loaded,
            &mut sm,
            &mut interner,
            &mut diags,
            &opts,
            emi_cache,
        ) {
            Ok(r) => r,
            Err(e) => {
                eprint!("{e}");
                return ExitCode::FAILURE;
            }
        };

        let out_path = match &out_opt {
            Some(o) if plan.targets.len() == 1 => o.clone(),
            Some(o) => o.join(&target.name),
            None => out_dir_path.join(&target.name),
        };
        let write_res = match target.kind {
            BuildTargetKind::Exe => write_executable_elf(&emitted.elf.bytes, &out_path),
            BuildTargetKind::Shared => write_shared_elf(&emitted.elf.bytes, &out_path),
        };
        if let Err(e) = write_res {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
        report_built_elf(&out_path, target.kind == BuildTargetKind::Shared, &emitted);
    }

    if audit_tcb {
        print!("{}", combined_report.format_report());
    }
    ExitCode::SUCCESS
}

fn cmd_run(args: &[String]) -> ExitCode {
    let mut explicit_native = false;
    let mut interp = false;
    let mut sanitize = false;
    let mut emi_cache: Option<PathBuf> = None;
    let mut needed_libs: Vec<String> = Vec::new();
    let mut inputs = Vec::new();
    let mut prog_args: Vec<String> = Vec::new();

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--" => {
                prog_args.extend_from_slice(&args[i + 1..]);
                break;
            }
            "--native" => explicit_native = true,
            "--interp" => interp = true,
            "--sanitize" => sanitize = true,
            "--emi-cache" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("--emi-cache requires a directory");
                    return ExitCode::FAILURE;
                }
                emi_cache = Some(PathBuf::from(&args[i]));
            }
            "-l" => {
                i += 1;
                if i >= args.len() {
                    eprintln!("-l requires a library name");
                    return ExitCode::FAILURE;
                }
                push_unique_lib(&mut needed_libs, resolve_lib_soname(&args[i]));
            }
            "-h" | "--help" => {
                eprintln!(
                    "usage: eonc run [--native | --interp] [--sanitize] [-l LIB] [--emi-cache DIR] <path...> [-- <prog_args...>]"
                );
                return ExitCode::SUCCESS;
            }
            other => {
                if let Some(lib) = other.strip_prefix("-l") {
                    if lib.is_empty() {
                        eprintln!("-l requires a library name");
                        return ExitCode::FAILURE;
                    }
                    push_unique_lib(&mut needed_libs, resolve_lib_soname(lib));
                } else if other.starts_with('-') {
                    eprintln!("unknown run flag `{other}`");
                    return ExitCode::FAILURE;
                } else {
                    inputs.push(PathBuf::from(other));
                }
            }
        }
        i += 1;
    }

    if explicit_native && interp {
        eprintln!("cannot combine --native and --interp");
        return ExitCode::FAILURE;
    }
    if inputs.is_empty() {
        eprintln!(
            "usage: eonc run [--native | --interp] [--sanitize] [-l LIB] [--emi-cache DIR] <path...> [-- <prog_args...>]"
        );
        return ExitCode::FAILURE;
    }

    let loaded = match load_eon_sources(&inputs) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("{e}");
            return ExitCode::FAILURE;
        }
    };
    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let mut diags = DiagBag::new();
    if interp {
        return match interpret_files_with_args(&loaded, &mut sm, &mut interner, &mut diags, &prog_args) {
            Ok(res) => {
                print!("{}", res.stdout);
                eprint!("{}", res.stderr);
                if res.exit_code == 0 {
                    ExitCode::SUCCESS
                } else {
                    ExitCode::from((res.exit_code.clamp(1, 255)) as u8)
                }
            }
            Err(e) => {
                eprint!("{e}");
                if !e.ends_with('\n') {
                    eprintln!();
                }
                ExitCode::FAILURE
            }
        };
    }

    let pid = std::process::id();
    let out_bin = env::temp_dir().join(format!("eonc_run_{pid}.bin"));
    let opts = X64Options {
        mode: ElfMode::LinuxPie,
        sanitize,
        needed_libs,
    };
    let emitted = match compile_files_to_x64_elf(
        &loaded,
        &mut sm,
        &mut interner,
        &mut diags,
        &opts,
        emi_cache.as_deref(),
    ) {
        Ok(r) => r,
        Err(e) => {
            eprint!("{e}");
            return ExitCode::FAILURE;
        }
    };
    if let Err(e) = write_executable_elf(&emitted.elf.bytes, &out_bin) {
        let _ = fs::remove_file(&out_bin);
        eprintln!("{e}");
        return ExitCode::FAILURE;
    }
    let run_res = run_compiled_binary_with_args(&out_bin, &prog_args);
    let _ = fs::remove_file(&out_bin);
    match run_res {
        Ok((code, stdout, stderr)) => {
            print!("{stdout}");
            eprint!("{stderr}");
            if code == 0 {
                ExitCode::SUCCESS
            } else {
                ExitCode::from((code.clamp(1, 255)) as u8)
            }
        }
        Err(e) => {
            eprintln!("{e}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_lib_soname;

    #[test]
    fn resolve_lib_soname_maps_short_and_verbatim_names() {
        assert_eq!(resolve_lib_soname("c"), "libc.so.6");
        assert_eq!(resolve_lib_soname("m"), "libm.so.6");
        assert_eq!(resolve_lib_soname("pthread"), "libpthread.so.6");
        assert_eq!(resolve_lib_soname("dl"), "libdl.so.6");
        assert_eq!(resolve_lib_soname("z"), "libz.so");
        assert_eq!(resolve_lib_soname("libc.so.6"), "libc.so.6");
        assert_eq!(resolve_lib_soname("libfoo.so"), "libfoo.so");
        assert_eq!(resolve_lib_soname("/usr/lib/libbar.so"), "/usr/lib/libbar.so");
    }
}
