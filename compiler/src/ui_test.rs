use std::fs;
use std::path::{Path, PathBuf};

use crate::diag::DiagBag;
use crate::span::{FileId, Interner, SourceMap};
use crate::{check_files_in_mem, collect_eon_files, parse_bytes};

#[derive(Clone, Debug)]
pub struct ExpectedError {
    pub line: u32,
    pub code: String,
    pub message_substr: Option<String>,
}

#[derive(Debug)]
pub struct UiFailure {
    pub path: PathBuf,
    pub reason: String,
}

#[derive(Debug)]
pub struct UiSummary {
    pub total: usize,
    pub passed: usize,
    pub failures: Vec<UiFailure>,
}

pub fn parse_expected_errors(text: &str) -> Vec<ExpectedError> {
    let mut expected = Vec::new();
    for (idx, raw_line) in text.lines().enumerate() {
        let line_no = (idx + 1) as u32;
        let Some(pos) = raw_line.find("//~") else {
            continue;
        };
        let mut rest = &raw_line[pos + 3..];
        let mut target_line = line_no;
        while let Some(r) = rest.strip_prefix('^') {
            target_line = target_line.saturating_sub(1);
            rest = r;
        }
        let rest = rest.trim();
        let Some(after_err) = rest.strip_prefix("ERROR") else {
            continue;
        };
        let after_err = after_err.trim();
        if after_err.is_empty() {
            continue;
        }
        let (code, substr) = match after_err.split_once(':') {
            Some((c, m)) => (c.trim().to_owned(), Some(m.trim().to_owned())),
            None => {
                let code_tok = after_err.split_whitespace().next().unwrap_or("");
                (code_tok.to_owned(), None)
            }
        };
        if !code.is_empty() {
            expected.push(ExpectedError {
                line: target_line,
                code,
                message_substr: substr.filter(|s| !s.is_empty()),
            });
        }
    }
    expected
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum UiMode {
    Parse,
    Check,
}

enum UiCase {
    Single { path: PathBuf, mode: UiMode },
    Package { dir: PathBuf, files: Vec<PathBuf> },
}

pub fn run_ui_dir(root: &Path) -> std::io::Result<UiSummary> {
    let mut cases = Vec::new();
    collect_ui_cases(root, &mut cases)?;

    let mut passed = 0usize;
    let mut failures = Vec::new();

    for case in &cases {
        let (display_path, res) = match case {
            UiCase::Single { path, mode } => match mode {
                UiMode::Parse => (path.clone(), check_single_parse_ui_file(path)),
                UiMode::Check => (path.clone(), check_single_semantic_ui_file(path)),
            },
            UiCase::Package { dir, files } => (dir.clone(), check_package_ui_dir(files)),
        };
        match res {
            Ok(()) => passed += 1,
            Err(reason) => failures.push(UiFailure {
                path: display_path,
                reason,
            }),
        }
    }

    Ok(UiSummary {
        total: cases.len(),
        passed,
        failures,
    })
}

fn dir_mode(path: &Path) -> Option<UiMode> {
    for comp in path.components() {
        let s = comp.as_os_str().to_string_lossy();
        if s.starts_with("check_") {
            return Some(UiMode::Check);
        }
        if s.starts_with("parse_") {
            return Some(UiMode::Parse);
        }
    }
    None
}

fn collect_ui_cases(dir: &Path, out: &mut Vec<UiCase>) -> std::io::Result<()> {
    if dir.is_file() {
        let mode = dir_mode(dir).unwrap_or(UiMode::Parse);
        out.push(UiCase::Single {
            path: dir.to_path_buf(),
            mode,
        });
        return Ok(());
    }
    let mut entries: Vec<_> = fs::read_dir(dir)?.collect::<Result<Vec<_>, _>>()?;
    entries.sort_by_key(|e| e.path());
    for entry in entries {
        let p = entry.path();
        if p.is_dir() {
            let name = p.file_name().and_then(|s| s.to_str()).unwrap_or("");
            if name.starts_with("pkg_") && dir_mode(&p) == Some(UiMode::Check) {
                let mut files = Vec::new();
                collect_eon_files(&p, &mut files)?;
                if !files.is_empty() {
                    out.push(UiCase::Package { dir: p, files });
                }
            } else {
                collect_ui_cases(&p, out)?;
            }
        } else if let Some(name) = p.file_name().and_then(|s| s.to_str())
            && name.ends_with(".eon")
        {
            let mode = dir_mode(&p).unwrap_or(UiMode::Parse);
            out.push(UiCase::Single { path: p, mode });
        }
    }
    Ok(())
}

pub fn check_single_ui_file(path: &Path) -> Result<(), String> {
    match dir_mode(path).unwrap_or(UiMode::Parse) {
        UiMode::Parse => check_single_parse_ui_file(path),
        UiMode::Check => check_single_semantic_ui_file(path),
    }
}

fn extract_expected_for_bytes(bytes: &[u8]) -> Vec<ExpectedError> {
    match std::str::from_utf8(bytes) {
        Ok(s) => parse_expected_errors(s),
        Err(_) => parse_expected_errors(&String::from_utf8_lossy(bytes)),
    }
}

fn check_single_parse_ui_file(path: &Path) -> Result<(), String> {
    let bytes = fs::read(path).map_err(|e| format!("failed to read file: {}", e))?;
    let expected = extract_expected_for_bytes(&bytes);
    let path_str = path.to_string_lossy().to_string();

    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let mut diags = DiagBag::new();
    parse_bytes(&path_str, bytes, &mut sm, &mut interner, &mut diags);

    let fid = FileId(0);
    let tagged: Vec<(FileId, ExpectedError)> =
        expected.into_iter().map(|e| (fid, e)).collect();
    verify_diagnostics(&sm, &diags, &tagged)
}

fn check_single_semantic_ui_file(path: &Path) -> Result<(), String> {
    let bytes = fs::read(path).map_err(|e| format!("failed to read file: {}", e))?;
    let expected = extract_expected_for_bytes(&bytes);
    let path_str = path.to_string_lossy().to_string();

    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let mut diags = DiagBag::new();
    let loaded = vec![(path_str, bytes)];
    let _ = check_files_in_mem(&loaded, &mut sm, &mut interner, &mut diags);

    let fid = FileId(0);
    let tagged: Vec<(FileId, ExpectedError)> =
        expected.into_iter().map(|e| (fid, e)).collect();
    verify_diagnostics(&sm, &diags, &tagged)
}

fn check_package_ui_dir(files: &[PathBuf]) -> Result<(), String> {
    let mut loaded = Vec::with_capacity(files.len());
    let mut tagged = Vec::new();
    for (idx, p) in files.iter().enumerate() {
        let bytes = fs::read(p)
            .map_err(|e| format!("failed to read `{}`: {}", p.display(), e))?;
        let fid = FileId(idx as u32);
        for exp in extract_expected_for_bytes(&bytes) {
            tagged.push((fid, exp));
        }
        loaded.push((p.to_string_lossy().to_string(), bytes));
    }

    let mut sm = SourceMap::new();
    let mut interner = Interner::new();
    let mut diags = DiagBag::new();
    let _ = check_files_in_mem(&loaded, &mut sm, &mut interner, &mut diags);

    verify_diagnostics(&sm, &diags, &tagged)
}

fn verify_diagnostics(
    sm: &SourceMap,
    diags: &DiagBag,
    expected: &[(FileId, ExpectedError)],
) -> Result<(), String> {
    let actual: Vec<(FileId, u32, String, String)> = diags
        .diagnostics()
        .iter()
        .map(|d| {
            let line = if d.span.file != FileId::DUMMY {
                sm.file(d.span.file).line_col(d.span.start).0
            } else {
                0
            };
            (d.span.file, line, d.code.to_string(), d.message.clone())
        })
        .collect();

    if expected.is_empty() {
        if !actual.is_empty() {
            return Err(format!(
                "expected 0 errors, got {}:\n{}",
                actual.len(),
                diags.render_all(sm)
            ));
        }
        return Ok(());
    }

    let mut matched_actual = vec![false; actual.len()];
    let mut missing = Vec::new();

    for (efid, exp) in expected {
        let found = actual.iter().enumerate().find(|(i, (fid, line, code, msg))| {
            !matched_actual[*i]
                && *fid == *efid
                && *line == exp.line
                && *code == exp.code
                && exp
                    .message_substr
                    .as_ref()
                    .map(|sub| msg.contains(sub))
                    .unwrap_or(true)
        });
        if let Some((i, _)) = found {
            matched_actual[i] = true;
        } else {
            missing.push(format!(
                "{} line {}: expected {} {:?}",
                sm.file(*efid).name,
                exp.line,
                exp.code,
                exp.message_substr
            ));
        }
    }

    let mut unexpected = Vec::new();
    for (i, (fid, line, code, msg)) in actual.iter().enumerate() {
        if !matched_actual[i] {
            let fname = if *fid != FileId::DUMMY {
                sm.file(*fid).name.as_str()
            } else {
                "<dummy>"
            };
            unexpected.push(format!("{} line {}: unexpected {} `{}`", fname, line, code, msg));
        }
    }

    if missing.is_empty() && unexpected.is_empty() {
        Ok(())
    } else {
        let mut msg = String::new();
        for m in missing {
            msg.push_str("  - missing: ");
            msg.push_str(&m);
            msg.push('\n');
        }
        for u in unexpected {
            msg.push_str("  - unexpected: ");
            msg.push_str(&u);
            msg.push('\n');
        }
        msg.push_str(&diags.render_all(sm));
        Err(msg)
    }
}
