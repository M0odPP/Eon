const vscode = require("vscode");
const cp = require("child_process");
const fs = require("fs");
const path = require("path");

let lspProc = null;
let nextId = 1;
const pending = new Map();
let recvBuf = Buffer.alloc(0);
let diagCollection = null;

function resolveCompilerPath() {
  const cfg = vscode.workspace.getConfiguration("eon");
  const configured = cfg.get("compilerPath", "eonc");
  if (configured && configured !== "eonc" && fs.existsSync(configured)) {
    return configured;
  }
  const folders = vscode.workspace.workspaceFolders || [];
  for (const f of folders) {
    const local = path.join(f.uri.fsPath, "selfhost", "out", "eonc2_rel");
    if (fs.existsSync(local)) {
      return local;
    }
  }
  return configured || "eonc";
}

function sendJsonRpc(msg) {
  if (!lspProc || !lspProc.stdin || lspProc.killed) return;
  const body = Buffer.from(JSON.stringify(msg), "utf8");
  const hdr = Buffer.from(`Content-Length: ${body.length}\r\n\r\n`, "utf8");
  lspProc.stdin.write(Buffer.concat([hdr, body]));
}

function request(method, params) {
  return new Promise((resolve) => {
    if (!lspProc || lspProc.killed) {
      resolve(null);
      return;
    }
    const id = nextId++;
    pending.set(id, resolve);
    sendJsonRpc({ jsonrpc: "2.0", id, method, params });
  });
}

function notify(method, params) {
  sendJsonRpc({ jsonrpc: "2.0", method, params });
}

function handleMessage(msg) {
  if (msg.id !== undefined && pending.has(msg.id)) {
    const fn = pending.get(msg.id);
    pending.delete(msg.id);
    fn(msg.result !== undefined ? msg.result : null);
    return;
  }
  if (msg.method === "textDocument/publishDiagnostics" && msg.params && diagCollection) {
    const uri = vscode.Uri.parse(msg.params.uri);
    const diags = (msg.params.diagnostics || []).map((d) => {
      const r = d.range || { start: { line: 0, character: 0 }, end: { line: 0, character: 1 } };
      const range = new vscode.Range(r.start.line, r.start.character, r.end.line, r.end.character);
      const sev =
        d.severity === 2
          ? vscode.DiagnosticSeverity.Warning
          : d.severity === 3
            ? vscode.DiagnosticSeverity.Information
            : vscode.DiagnosticSeverity.Error;
      const item = new vscode.Diagnostic(range, d.message || "", sev);
      item.source = "eonc";
      return item;
    });
    diagCollection.set(uri, diags);
  }
}

function onStdoutData(chunk) {
  recvBuf = Buffer.concat([recvBuf, chunk]);
  while (true) {
    const sep = recvBuf.indexOf("\r\n\r\n");
    if (sep < 0) break;
    const hdr = recvBuf.slice(0, sep).toString("ascii");
    const m = /Content-Length:\s*(\d+)/i.exec(hdr);
    if (!m) {
      recvBuf = recvBuf.slice(sep + 4);
      continue;
    }
    const len = parseInt(m[1], 10);
    if (recvBuf.length < sep + 4 + len) break;
    const body = recvBuf.slice(sep + 4, sep + 4 + len).toString("utf8");
    recvBuf = recvBuf.slice(sep + 4 + len);
    try {
      handleMessage(JSON.parse(body));
    } catch (_) {}
  }
}

function startLsp(context) {
  const cfg = vscode.workspace.getConfiguration("eon");
  if (!cfg.get("lsp.enable", true)) return;
  const bin = resolveCompilerPath();
  try {
    lspProc = cp.spawn(bin, ["lsp"], { stdio: ["pipe", "pipe", "ignore"] });
  } catch (_) {
    lspProc = null;
    return;
  }
  lspProc.stdout.on("data", onStdoutData);
  lspProc.on("exit", () => {
    lspProc = null;
  });

  const rootUri =
    vscode.workspace.workspaceFolders && vscode.workspace.workspaceFolders.length > 0
      ? vscode.workspace.workspaceFolders[0].uri.toString()
      : null;
  request("initialize", { processId: process.pid, rootUri, capabilities: {} }).then(() => {
    notify("initialized", {});
    for (const doc of vscode.workspace.textDocuments) {
      if (doc.languageId === "eon") {
        notify("textDocument/didOpen", {
          textDocument: { uri: doc.uri.toString(), languageId: "eon", version: doc.version, text: doc.getText() },
        });
      }
    }
  });

  context.subscriptions.push(
    vscode.workspace.onDidOpenTextDocument((doc) => {
      if (doc.languageId === "eon") {
        notify("textDocument/didOpen", {
          textDocument: { uri: doc.uri.toString(), languageId: "eon", version: doc.version, text: doc.getText() },
        });
      }
    }),
    vscode.workspace.onDidChangeTextDocument((ev) => {
      if (ev.document.languageId === "eon") {
        notify("textDocument/didChange", {
          textDocument: { uri: ev.document.uri.toString(), version: ev.document.version },
          contentChanges: [{ text: ev.document.getText() }],
        });
      }
    }),
    vscode.workspace.onDidCloseTextDocument((doc) => {
      if (doc.languageId === "eon") {
        notify("textDocument/didClose", { textDocument: { uri: doc.uri.toString() } });
      }
    }),
    vscode.languages.registerHoverProvider("eon", {
      async provideHover(doc, pos) {
        const res = await request("textDocument/hover", {
          textDocument: { uri: doc.uri.toString() },
          position: { line: pos.line, character: pos.character },
        });
        if (!res || !res.contents) return null;
        const val = typeof res.contents === "string" ? res.contents : res.contents.value || "";
        return val ? new vscode.Hover(new vscode.MarkdownString(val)) : null;
      },
    }),
    vscode.languages.registerDefinitionProvider("eon", {
      async provideDefinition(doc, pos) {
        const res = await request("textDocument/definition", {
          textDocument: { uri: doc.uri.toString() },
          position: { line: pos.line, character: pos.character },
        });
        if (!res) return null;
        const loc = Array.isArray(res) ? res[0] : res;
        if (!loc || !loc.uri || !loc.range) return null;
        return new vscode.Location(
          vscode.Uri.parse(loc.uri),
          new vscode.Range(loc.range.start.line, loc.range.start.character, loc.range.end.line, loc.range.end.character)
        );
      },
    }),
    vscode.languages.registerCompletionItemProvider("eon", {
      async provideCompletionItems(doc, pos) {
        const res = await request("textDocument/completion", {
          textDocument: { uri: doc.uri.toString() },
          position: { line: pos.line, character: pos.character },
        });
        const items = Array.isArray(res) ? res : (res && res.items) || [];
        return items.map((it) => new vscode.CompletionItem(it.label, vscode.CompletionItemKind.Keyword));
      },
    }),
    vscode.languages.registerDocumentFormattingEditProvider("eon", {
      async provideDocumentFormattingEdits(doc) {
        const edits = await request("textDocument/formatting", {
          textDocument: { uri: doc.uri.toString() },
          options: { tabSize: 4, insertSpaces: true },
        });
        if (!Array.isArray(edits)) return [];
        return edits.map((e) =>
          vscode.TextEdit.replace(
            new vscode.Range(e.range.start.line, e.range.start.character, e.range.end.line, e.range.end.character),
            e.newText || ""
          )
        );
      },
    })
  );
}

function activate(context) {
  diagCollection = vscode.languages.createDiagnosticCollection("eon");
  context.subscriptions.push(diagCollection);
  startLsp(context);
}

function deactivate() {
  if (lspProc && !lspProc.killed) {
    try {
      notify("shutdown", null);
      notify("exit", null);
      lspProc.kill();
    } catch (_) {}
  }
}

module.exports = { activate, deactivate };
