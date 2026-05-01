"""Convert captured ANSI output into HTML for embedding in MkDocs pages.

We don't take a runtime dep on ansi2html — `display.py` only emits a small,
predictable subset of SGR codes. Maps each code to an inline-styled <span>.

Usage:
    python docs/_scripts/render_ansi.py < input.ansi > output.html

The output is a self-contained `<pre class="terminal">…</pre>` block that
relies on `docs/stylesheets/terminal.css` for the frame styling.
"""
import html
import re
import sys

# (foreground, background, bold) → CSS class name suffix.
# These pair with rules in docs/stylesheets/terminal.css.
_CODE_TO_CLASS = {
    "1":     ("bold",),
    "2":     ("dim",),
    "31":    ("fg-red",),
    "32":    ("fg-green",),
    "33":    ("fg-yellow",),
    "34":    ("fg-blue",),
    "35":    ("fg-magenta",),
    "36":    ("fg-cyan",),
    "37":    ("fg-white",),
    "90":    ("fg-gray",),
    "47":    ("bg-highlight",),
    "1;32":  ("bold", "fg-green"),
    "1;37":  ("bold", "fg-white"),
}

_SGR_RE = re.compile(r"\x1b\[([0-9;]*)m")


def render(text: str) -> str:
    out = []
    pos = 0
    open_span = False
    for m in _SGR_RE.finditer(text):
        out.append(html.escape(text[pos:m.start()]))
        code = m.group(1)
        if open_span:
            out.append("</span>")
            open_span = False
        if code in ("", "0"):
            pass  # reset only — already closed above.
        elif code in _CODE_TO_CLASS:
            classes = " ".join(f"ansi-{c}" for c in _CODE_TO_CLASS[code])
            out.append(f'<span class="{classes}">')
            open_span = True
        # Unknown codes are silently dropped — better than corrupt output.
        pos = m.end()
    out.append(html.escape(text[pos:]))
    if open_span:
        out.append("</span>")
    return f'<pre class="terminal"><code>{"".join(out)}</code></pre>\n'


if __name__ == "__main__":
    sys.stdout.write(render(sys.stdin.read()))
