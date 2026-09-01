#!/usr/bin/env python3
"""Static scope checker for the Jinja-templated frontend JS.

Extracts every inline <script> block from templates/index.html and its
{% include %}'d tab files, in real document order, strips any {%- raw %}
wrapper markers, and runs the result through ESLint's no-undef rule to
catch "variable/function referenced but never in scope" bugs — the exact
class of bug that broke the TradeAI price-refresh loop three times in a
row this session (dreamScanning, activeTab, updateMacroStatusBadge).

A reference guarded by `typeof x === 'function'` is reported separately
and does not fail the run — that pattern never throws at runtime even
when x is genuinely undeclared, so it's a legitimate feature-detection
idiom used a few times in this codebase, not a bug.

Requires Node.js (for `npx eslint`) on PATH. No project dependency is
installed; npx fetches/caches eslint itself on first run.

Usage:  python check_js.py
Exit codes: 0 = clean, 1 = real issues found, 2 = tool/setup problem.
"""
import json
import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.abspath(__file__))
TEMPLATES = os.path.join(ROOT, "templates")
INDEX_HTML = os.path.join(TEMPLATES, "index.html")

SCRIPT_RE = re.compile(r"<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)</script>", re.IGNORECASE)
INCLUDE_RE = re.compile(r"\{%-?\s*include\s+'([^']+)'\s*-?%\}")
RAW_RE = re.compile(r"\{%-?\s*raw\s*-?%\}([\s\S]*?)\{%-?\s*endraw\s*-?%\}")
TYPEOF_GUARD_TMPL = r"typeof\s+{}\s*(===|==)\s*['\"]function['\"]"
NOT_DEFINED_RE = re.compile(r"'([^']+)' is not defined")

# A deliberately curated (not exhaustive) set of standard browser/JS globals
# this codebase actually uses. Anything missing here shows up as a reported
# issue that a human can recognize as a real global and add to the list —
# a small maintenance cost, much cheaper than vendoring ESLint's full
# multi-hundred-entry browser global list for APIs this app will never call.
BROWSER_GLOBALS = [
    "window", "document", "console", "fetch", "localStorage", "sessionStorage",
    "setTimeout", "setInterval", "clearTimeout", "clearInterval",
    "alert", "confirm", "prompt", "navigator", "location", "history",
    "URL", "URLSearchParams", "globalThis", "self", "top", "parent", "frames", "opener",
    "Date", "Math", "JSON", "Promise", "Array", "Object", "String", "Number", "Boolean",
    "RegExp", "Error", "Map", "Set", "WeakMap", "WeakSet", "Symbol",
    "encodeURIComponent", "decodeURIComponent", "encodeURI", "decodeURI",
    "isNaN", "isFinite", "parseFloat", "parseInt",
    "requestAnimationFrame", "cancelAnimationFrame", "requestIdleCallback", "cancelIdleCallback",
    "performance", "crypto", "Intl", "structuredClone", "queueMicrotask",
    "Blob", "File", "FileReader", "FormData", "Headers", "Request", "Response",
    "XMLHttpRequest", "WebSocket", "Worker", "Image", "Audio",
    "CustomEvent", "Event", "EventTarget", "MutationObserver", "ResizeObserver", "IntersectionObserver",
    "screen", "innerWidth", "innerHeight", "outerWidth", "outerHeight",
    "scrollX", "scrollY", "pageXOffset", "pageYOffset", "devicePixelRatio",
    "open", "close", "print", "focus", "blur", "moveTo", "moveBy", "resizeTo", "resizeBy",
    "scroll", "scrollBy", "scrollTo", "getComputedStyle", "getSelection", "matchMedia",
    "postMessage", "atob", "btoa", "addEventListener", "removeEventListener", "dispatchEvent",
    "TextEncoder", "TextDecoder", "AbortController", "AbortSignal",
    "ReadableStream", "WritableStream", "indexedDB", "IDBKeyRange", "Notification", "caches", "CSS",
    "DOMParser", "XMLSerializer", "HTMLElement", "Node", "NodeList", "Element", "Document",
    "DocumentFragment", "ShadowRoot", "Range", "Selection", "DOMRect", "DOMRectReadOnly",
    "MouseEvent", "KeyboardEvent", "PointerEvent", "TouchEvent", "WheelEvent", "FocusEvent",
    "InputEvent", "DragEvent", "ClipboardEvent", "StorageEvent", "PopStateEvent", "HashChangeEvent",
    "ErrorEvent", "ProgressEvent", "MessageEvent", "CloseEvent", "name", "length", "status",
]

ESLINT_CONFIG = """export default [
  {{
    files: ["**/*.js"],
    languageOptions: {{
      ecmaVersion: "latest",
      sourceType: "script",
      globals: {{
{globals}
      }},
    }},
    rules: {{ "no-undef": "error" }},
  }},
];
""".format(globals="\n".join(f'        "{g}": "writable",' for g in BROWSER_GLOBALS))


def extract_blocks(path):
    """Every <script> tag's JS in this file, in order, {%- raw %} markers stripped."""
    with open(path, encoding="utf-8") as f:
        content = f.read()
    blocks = []
    for m in SCRIPT_RE.finditer(content):
        body = m.group(1)
        raw_match = RAW_RE.search(body)
        blocks.append(raw_match.group(1) if raw_match else body)
    return blocks


def document_order_files():
    """index.html itself, then each {% include %} target, in the order
    Jinja actually assembles them — matters because a variable/function
    declared in one included file may legitimately be used in one included
    later, but not the other way around."""
    files = [INDEX_HTML]
    with open(INDEX_HTML, encoding="utf-8") as f:
        content = f.read()
    for m in INCLUDE_RE.finditer(content):
        files.append(os.path.join(TEMPLATES, m.group(1)))
    return files


def build_combined_js():
    combined_lines = []
    line_map = []  # (start_line_in_combined, source_file_relpath, block_index)
    for path in document_order_files():
        rel = os.path.relpath(path, ROOT)
        if not os.path.exists(path):
            print(f"WARN: {rel} not found, skipping", file=sys.stderr)
            continue
        for i, block in enumerate(extract_blocks(path)):
            line_map.append((len(combined_lines) + 1, rel, i))
            combined_lines.append(f"// ===== {rel} block {i} =====")
            combined_lines.extend(block.split("\n"))
    return "\n".join(combined_lines), combined_lines, line_map


def locate(line_no, line_map):
    best = ("?", -1)
    for start_line, rel, block_idx in line_map:
        if start_line <= line_no:
            best = (rel, block_idx)
        else:
            break
    return best


def run_eslint(combined_js):
    with tempfile.TemporaryDirectory() as tmp:
        combined_path = os.path.join(tmp, "combined.js")
        config_path = os.path.join(tmp, "eslint.config.mjs")
        with open(combined_path, "w", encoding="utf-8") as f:
            f.write(combined_js)
        with open(config_path, "w", encoding="utf-8") as f:
            f.write(ESLINT_CONFIG)
        try:
            result = subprocess.run(
                ["npx", "--yes", "eslint", "--config", "eslint.config.mjs", "--format", "json", "combined.js"],
                capture_output=True, text=True, encoding="utf-8", timeout=120, cwd=tmp,
                shell=(os.name == "nt"),
            )
        except FileNotFoundError:
            print("ERROR: npx/node not found on PATH — install Node.js to run this check.", file=sys.stderr)
            return None
        except subprocess.TimeoutExpired:
            print("ERROR: eslint timed out.", file=sys.stderr)
            return None
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError:
            print("ERROR: could not parse eslint output:", file=sys.stderr)
            print(result.stdout or "(empty stdout)", file=sys.stderr)
            print(result.stderr, file=sys.stderr)
            return None


def main():
    combined_js, combined_lines, line_map = build_combined_js()
    report = run_eslint(combined_js)
    if report is None:
        return 2

    messages = [msg for file_report in report for msg in file_report.get("messages", [])]

    real_issues = []
    safe_typeof_guards = []
    for msg in messages:
        line_no = msg.get("line", 0)
        line_text = combined_lines[line_no - 1] if 0 < line_no <= len(combined_lines) else ""
        rel, block_idx = locate(line_no, line_map)
        var_match = NOT_DEFINED_RE.search(msg.get("message", ""))
        var_name = var_match.group(1) if var_match else None
        is_typeof_guard = bool(var_name) and re.search(TYPEOF_GUARD_TMPL.format(re.escape(var_name)), line_text)
        entry = {"file": rel, "block": block_idx, "text": line_text.strip(), "message": msg.get("message", "")}
        (safe_typeof_guards if is_typeof_guard else real_issues).append(entry)

    n_files = len({rel for _, rel, _ in line_map})
    print(f"Scanned {n_files} template files, {len(line_map)} <script> blocks, {len(combined_lines)} lines of JS.\n")

    if safe_typeof_guards:
        print(f"Ignored {len(safe_typeof_guards)} safe typeof-guarded reference(s) (never throws at runtime):")
        for e in safe_typeof_guards:
            print(f"  {e['file']} (block {e['block']}): {e['text']}")
        print()

    if not real_issues:
        print("No undefined-variable issues found.")
        return 0

    print(f"Found {len(real_issues)} potential issue(s):\n")
    for e in real_issues:
        print(f"  {e['file']} (block {e['block']}): {e['message']}")
        print(f"    {e['text']}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
