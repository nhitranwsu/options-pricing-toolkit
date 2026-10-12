"""Presentation layer for the options toolkit: theme, HTML panels and Plotly figures.

Everything here returns strings or figures, so app.py stays about inputs and layout only.
Colors follow one rule everywhere: blue is long or positive (profit, buy side), red is short or negative (loss, sell side),
amber marks the passage of time, green and red text only ever mean gain and loss and always come with a sign.
Shorter expiries are drawn as an ordinal blue ramp against the neutral current expiry.
"""
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytics import moneyness, expected_move, breakevens

C = dict(
    bg="#0d0d0d", panel="#1a1a19", panel2="#222221", line="#2c2c2a", axis="#383835",
    ink="#f0efec", ink2="#c3c2b7", muted="#898781",
    long="#3987e5", short="#e66767", time="#c98500", vol="#9085e9",
    gain="#0ca30c", loss="#e66767", aqua="#199e70",
    ramp1="#86b6ef", ramp2="#3987e5",
)
FONT = "IBM Plex Sans, system-ui, -apple-system, Segoe UI, sans-serif"

_CSS_BODY = """
.stApp { background: var(--bg); color: var(--ink); }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbarActions"], [data-testid="stMainMenu"], [data-testid="stAppDeployButton"], [data-testid="stStatusWidget"], [data-testid="stDecoration"], #MainMenu, footer { display: none !important; }
[data-testid="stExpandSidebarButton"] { color: var(--ink2) !important; }
.block-container { padding: 1.2rem 1.6rem 2.5rem; max-width: 100%; }
[data-testid="stSidebar"] { background: var(--panel); border-right: 1px solid var(--line); }
[data-testid="stSidebar"] label p, [data-testid="stSidebar"] .stMarkdown p { font-size: 13px; }
[data-testid="stPlotlyChart"] { border: 1px solid var(--line); border-radius: 6px; overflow: hidden; background: var(--panel); }

.tm, .tm * { font-family: 'IBM Plex Sans', system-ui, -apple-system, 'Segoe UI', sans-serif; box-sizing: border-box; }
.tm { color: var(--ink); font-size: 14px; font-variant-numeric: tabular-nums; }
.gain { color: var(--gain); }
.loss { color: var(--loss); }
.mu { color: var(--muted); }

/* top bar */
.brand { display: flex; align-items: baseline; gap: 14px; padding: 12px 16px; background: var(--panel);
         border: 1px solid var(--line); border-radius: 6px; }
.brand b { font-size: 19px; font-weight: 600; letter-spacing: -0.01em; white-space: nowrap; }
.brand span { color: var(--ink2); font-size: 13px; }

/* status strip */
.meta { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 24px; padding: 10px 2px 12px; font-size: 13px;
        border-bottom: 1px solid var(--line); margin-bottom: 12px; }
.meta .kv em { font-style: normal; color: var(--muted); margin-right: 7px; }
.pill { display: inline-flex; align-items: center; gap: 8px; padding: 3px 11px 3px 9px; border-radius: 999px;
        border: 1px solid var(--axis); font-weight: 600; font-size: 12px; color: var(--ink); }
.pill i { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
.pill.long i { background: var(--long); }
.pill.short i { background: var(--short); }

/* chart legend, rendered as HTML so it never wraps over the panel labels */
.leg { display: flex; flex-wrap: wrap; gap: 4px 18px; align-items: center; background: var(--panel); border: 1px solid var(--line);
       border-radius: 6px; padding: 8px 14px; margin: 0; color: var(--ink2); font-size: 12px; }
.leg span { display: inline-flex; align-items: center; gap: 7px; white-space: nowrap; }
.leg i { display: inline-block; width: 16px; height: 0; border-top: 2px solid; }
.leg i.dash { border-top-style: dashed; }
.leg i.dot { border-top-style: dotted; }

/* KPI strip */
.kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 1px; background: var(--line);
        border: 1px solid var(--line); border-radius: 6px; overflow: hidden; margin-bottom: 12px; }
.kpi { background: var(--panel); padding: 11px 15px 12px; min-width: 0; }
.kpi .l { color: var(--muted); font-size: 12px; }
.kpi .v { font-size: 27px; font-weight: 600; line-height: 1.2; margin-top: 3px; white-space: nowrap; }
.kpi .s { color: var(--ink2); font-size: 12px; line-height: 1.35; margin-top: 3px; }

/* section titles */
.pt { color: var(--ink2); font-size: 13px; font-weight: 500; margin: 0 0 6px; }
.pt.sp { margin-top: 14px; }
.note { color: var(--muted); font-size: 12px; margin-top: 6px; margin-bottom: 8px; line-height: 1.45; }

/* key value tables */
.kvt { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
.kvt table { width: 100%; border-collapse: collapse; font-size: 13px; }
.tm table th, .tm table td { border-left: 0; border-right: 0; border-top: 0; background: transparent; }
.kvt td { padding: 4px 10px; min-height: 28px; border-bottom: 1px solid var(--line); line-height: 1.3; }
.kvt tr:last-child td { border-bottom: 0; }
.kvt td:first-child { color: var(--ink2); white-space: normal; }
.kvt td:last-child { text-align: right; font-weight: 500; white-space: nowrap; }
.kvt tr.tot td { border-top: 1px solid var(--axis); font-weight: 600; }
.kvt tr.tot td:first-child { color: var(--ink); }

/* generic data tables */
.blot { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; overflow: hidden; }
.blot table { width: 100%; border-collapse: collapse; font-size: 13px; }
.blot th { color: var(--muted); font-weight: 500; font-size: 12px; text-align: right; padding: 7px 10px; border-bottom: 1px solid var(--line); }
.blot td { text-align: right; padding: 0 10px; height: 27px; border-bottom: 1px solid var(--line); white-space: nowrap; }
.blot tr:last-child td { border-bottom: 0; }
.blot th:first-child, .blot td:first-child { text-align: left; }
.blot.txt th, .blot.txt td { text-align: left; vertical-align: top; height: auto; padding: 10px 14px; line-height: 1.5; font-size: 13px; white-space: normal; }
.blot.txt td:first-child { color: var(--ink); font-weight: 500; white-space: nowrap; }
.blot.txt td { color: var(--ink2); }

/* option chain */
.chain th.grp { text-align: center; color: var(--ink2); font-weight: 600; letter-spacing: .04em; border-bottom: 1px solid var(--axis); }
.chain td.k { text-align: center; font-weight: 600; background: var(--panel2); color: var(--ink); }
.chain td.mid { color: var(--ink2); background: rgba(240,239,236,.03); }
.chain td.itm { background: rgba(57,135,229,.10); }
.chain tr.atm td { border-top: 1px solid var(--ink2); border-bottom: 1px solid var(--ink2); }
.chain th.k { text-align: center; }
.sep { border-left: 1px solid var(--axis); }

/* solver result */
.res { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 12px 16px; display: flex;
       justify-content: space-between; align-items: baseline; gap: 16px; margin-top: 8px; }
.res .l { color: var(--muted); font-size: 12px; }
.res .v { font-size: 26px; font-weight: 600; line-height: 1.2; }
.res .s { color: var(--ink2); font-size: 12px; text-align: right; }
.res.err { border-color: var(--short); }

/* author and copyright */
.author { padding: 10px 13px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel2); font-size: 12px;
          color: var(--ink2); line-height: 1.45; margin-bottom: 14px; }
.author b { display: block; color: var(--ink); font-weight: 600; font-size: 14px; letter-spacing: -0.01em; }
.foot { margin-top: 30px; padding-top: 12px; border-top: 1px solid var(--line); color: var(--muted); font-size: 12px; line-height: 1.5; }
"""


def css():
    root = ":root{" + "".join(f"--{k}:{v};" for k, v in C.items()) + "}"
    font = "@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');"
    return "<style>" + font + root + _CSS_BODY + "</style>"


# ------------------------------------------------------------------------------------------------- helpers
def _sg(x):
    return "gain" if x >= 0 else "loss"


def _rgba(h, a):
    h = h.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{a})"


def usd(x, sign=True, dec=2):
    s = f"{abs(x):,.{dec}f}"
    if sign:
        return f"{'+' if x >= 0 else '−'}${s}"
    return f"{'−' if x < 0 else ''}${s}"


def _n(x, spec):
    """Signed number with a true minus sign, e.g. _n(-1.5, ".2f") gives '−1.50'."""
    return format(x, "+" + spec).replace("-", "−")


def sgn(x, dec=4):
    return f"{'+' if x >= 0 else '−'}{abs(x):.{dec}f}"


def _dec(price):
    return 2 if abs(price) >= 1.0 else 4


def _px(x, dec=None):
    d = _dec(x) if dec is None else dec
    return f"{x:,.{d}f}"


def _colored(x, text):
    return f'<span class="{_sg(x)}">{text}</span>'


def expiry_label(c):
    if c.dpy == 252.0:
        return f"{c.T * 252.0 * 6.5:.2f} trading hrs"
    return f"{c.T * 365.0:.1f} days"


# ------------------------------------------------------------------------------------------------- HTML panels


def author_html():
    return ('<div class="tm author"><b>Nhi Q Tran</b>'
            'Designed and built by Nhi Q Tran. Source code is private.</div>')


def foot_html():
    return ('<div class="tm foot">&copy; 2026 Nhi Q Tran. All rights reserved. '
            'The source code is private and may not be copied or redistributed. '
            'Research and education tool, not investment advice.</div>')


def brand_html():
    return ('<div class="tm brand"><b>Options pricer</b>'
            '<span>Black-Scholes-Merton, binomial tree, Greeks, scenario P&amp;L</span></div>')


def meta_html(c):
    label, pct = moneyness(c)
    mv, mvp = expected_move(c)
    side = "long" if c.qty > 0 else "short"
    mny = label if label == "ATM" else f"{label} {abs(pct):.1f}%"
    return (f'<div class="tm meta"><span class="pill {side}"><i></i>{side.upper()}</span>'
            f'<span class="kv"><em>Position</em>{abs(c.qty)} × {c.mult}</span>'
            f'<span class="kv"><em>Type</em>{c.option_type.capitalize()}</span>'
            f'<span class="kv"><em>Spot</em>{c.S:,.2f}</span>'
            f'<span class="kv"><em>Strike</em>{c.K:,.2f}</span>'
            f'<span class="kv"><em>Expiry</em>{expiry_label(c)} (T = {c.T:.4f}y)</span>'
            f'<span class="kv"><em>Vol</em>{c.sigma * 100:.2f}%</span>'
            f'<span class="kv"><em>Rate</em>{c.r * 100:.2f}%</span>'
            f'<span class="kv"><em>Div yield</em>{c.q * 100:.2f}%</span>'
            f'<span class="kv"><em>Moneyness</em>{mny}</span>'
            f'<span class="kv"><em>1σ move</em>±{mv:,.2f} ({mvp:.1f}%)</span></div>')


def _tile(label, value, sub=""):
    return f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div><div class="s">{sub}</div></div>'


def kpi_html(c, d):
    p = d["pos"]
    flat = "no position"
    tiles = [
        _tile("Option value", f'${d["price"]:,.4f}',
              f'intrinsic {d["intrinsic"]:.4f}, time value {d["time_value"]:.4f}'),
        _tile("Delta", sgn(d["delta"]),
              f'{_n(p["delta_sh"], ",.1f")} shares, {usd(p["delta_usd"], dec=0)} notional' if c.qty else flat),
        _tile("Gamma", f'{d["gamma"]:.4f}',
              f'{_n(p["gamma_1pct_sh"], ",.2f")} shares of delta per +1%' if c.qty else flat),
        _tile("Theta per day", sgn(d["theta_day"]),
              f'{_colored(p["theta_day"], usd(p["theta_day"]))} per day, position' if c.qty else flat),
        _tile("Vega per vol point", sgn(d["vega_pt"]),
              f'{_colored(p["vega_pt"], usd(p["vega_pt"]))} per vol pt, position' if c.qty else flat),
        _tile("Rho per 1% rate", sgn(d["rho_pct"]),
              f'{_colored(p["rho_pct"], usd(p["rho_pct"]))} per 1%, position' if c.qty else flat),
        _tile("Probability in the money", f'{d["prob_itm"] * 100:.1f}%', "risk neutral, N(d2)"),
        _tile("Breakeven at expiry", f'{d["breakeven"]:,.2f}',
              f'{_n(d["breakeven_pct"], ".1f")}% from spot, before carry'),
    ]
    return '<div class="tm kpis">' + "".join(tiles) + "</div>"


def kv_html(rows, total_last=False):
    body = ""
    for i, (k, v) in enumerate(rows):
        cls = ' class="tot"' if (total_last and i == len(rows) - 1) else ""
        body += f"<tr{cls}><td>{k}</td><td>{v}</td></tr>"
    return f'<div class="tm kvt"><table>{body}</table></div>'


def position_html(c, d, max_loss_value):
    p = d["pos"]
    if not c.qty:
        return kv_html([("Position", "flat")])
    ml = "Unlimited" if max_loss_value is None else usd(-max_loss_value)
    rows = [
        ("Position", f'{"Long" if c.qty > 0 else "Short"} {abs(c.qty)} × {c.mult}'),
        ("Market value", usd(p["value"], sign=False)),
        ("Delta, shares", f'{_n(p["delta_sh"], ",.1f")}'),
        ("Delta, notional", usd(p["delta_usd"], dec=0)),
        ("Gamma, Δ per +1% spot", f'{_n(p["gamma_1pct_sh"], ",.2f")}'),
        ("Theta, per day", _colored(p["theta_day"], usd(p["theta_day"]))),
        ("Vega, per vol pt", _colored(p["vega_pt"], usd(p["vega_pt"]))),
        ("Rho, per 1% rate", _colored(p["rho_pct"], usd(p["rho_pct"]))),
        ("Largest loss", f'<span class="loss">{ml}</span>'),
    ]
    return kv_html(rows)


def model_check_html(bs, eu, am, gap):
    def row(a, b=""):
        return (a, b)
    eu_txt = "n/a" if eu is None else f'{_px(eu, 4)} <span class="mu">({_n(eu - bs, ".4f")})</span>'
    am_txt = "n/a" if am is None else _px(am, 4)
    prem = "n/a" if (am is None or eu is None) else f'{am - eu:.4f}'
    rows = [
        row("Black-Scholes", _px(bs, 4)),
        row("Tree, European (N=200)", eu_txt),
        row("Tree, American (N=200)", am_txt),
        row("Early exercise premium", prem),
        row("Put-call parity gap", f'{gap:.1e}'.replace('-', '−')),
    ]
    return kv_html(rows)


def legend_html(items):
    """items: (kind, color, label) with kind in line, dash, dot."""
    out = ""
    for kind, color, label in items:
        cls = "" if kind == "line" else kind
        out += f'<span><i class="{cls}" style="border-color:{color}"></i>{label}</span>'
    return '<div class="tm leg">' + out + "</div>"


def explain_table_html(ex):
    rows = [(k, _colored(v, usd(v))) for k, v in ex["parts"].items()]
    rows.append(("Higher order terms", _colored(ex["residual"], usd(ex["residual"]))))
    rows.append(("Total, full revaluation", _colored(ex["full"], usd(ex["full"]))))
    return kv_html(rows, total_last=True)


def chain_html(c, ch):
    K = ch["K"]
    call, put = ch["call"], ch["put"]
    dec = 2 if float(np.max(call["price"])) >= 1.0 else 4
    atm = int(np.argmin(np.abs(K - c.S)))
    head = ('<tr><th class="grp" colspan="3">CALLS</th><th class="grp sep" colspan="4">STRIKE</th>'
            '<th class="grp sep" colspan="3">PUTS</th></tr>'
            '<tr><th>Price</th><th>Delta</th><th>Theta/day</th>'
            '<th class="sep k">Strike</th><th>IV</th><th>Gamma</th><th>Vega/pt</th>'
            '<th class="sep">Delta</th><th>Theta/day</th><th>Price</th></tr>')
    body = ""
    for i, k in enumerate(K):
        call_itm = " itm" if k < c.S else ""
        put_itm = " itm" if k > c.S else ""
        row_cls = ' class="atm"' if i == atm else ""
        body += (f'<tr{row_cls}>'
                 f'<td class="{call_itm.strip()}">{call["price"][i]:,.{dec}f}</td>'
                 f'<td class="{call_itm.strip()}">{_n(call["delta"][i], ".3f")}</td>'
                 f'<td class="{call_itm.strip()}">{_n(call["theta"][i] / c.dpy, ".4f")}</td>'
                 f'<td class="k sep">{k:,.2f}</td>'
                 f'<td class="mid">{ch["iv"][i] * 100:.2f}%</td>'
                 f'<td class="mid">{call["gamma"][i]:.4f}</td>'
                 f'<td class="mid">{call["vega"][i] / 100.0:.4f}</td>'
                 f'<td class="sep{put_itm}">{_n(put["delta"][i], ".3f")}</td>'
                 f'<td class="{put_itm.strip()}">{_n(put["theta"][i] / c.dpy, ".4f")}</td>'
                 f'<td class="{put_itm.strip()}">{put["price"][i]:,.{dec}f}</td></tr>')
    return f'<div class="tm blot chain"><table>{head}{body}</table></div>'


def iv_result_html(iv, vol_in, ok=True, msg=""):
    if not ok:
        return f'<div class="tm res err"><div><div class="l">Implied volatility</div><div class="v">n/a</div></div><div class="s">{msg}</div></div>'
    diff = (iv - vol_in) * 100.0
    return (f'<div class="tm res"><div><div class="l">Implied volatility</div><div class="v">{iv * 100:.4f}%</div></div>'
            f'<div class="s">{_n(diff, ".4f")} vol points<br>against the input vol</div></div>')


def xcheck_html(rows):
    """rows: (name, analytical, numerical, tolerance)."""
    head = "<tr><th>Greek</th><th>Analytical</th><th>Finite diff</th><th>Diff</th><th>OK</th></tr>"
    body = ""
    for name, a, n, tol in rows:
        diff = abs(a - n)
        mark = '<span class="gain">✓</span>' if diff <= tol else '<span class="loss">✗</span>'
        body += (f"<tr><td>{name}</td><td>{format(a, '.4f').replace('-', '−')}</td>"
                 f"<td>{format(n, '.4f').replace('-', '−')}</td><td>{diff:.0e}</td><td>{mark}</td></tr>")
    return f'<div class="tm blot"><table>{head}{body}</table></div>'


def assumptions_html():
    rows = [
        ("Volatility", "One flat σ for every strike and expiry. The Chain view adds an illustrative smile.",
         "A live surface per expiry, how skew moves with spot (sticky strike against sticky delta), term structure."),
        ("Exercise style", "Black-Scholes is European. The American tree is shown next to it for comparison.",
         "Listed single stock options are American. Early exercise matters for puts and for calls before a dividend."),
        ("Dividends and rates", "A continuous dividend yield and one flat rate.",
         "Discrete dividends, borrow cost, and a proper rate curve."),
        ("Greeks", "Analytical delta, gamma, vega, theta and rho. Cross Greeks (vanna, volga, charm) are not shown.",
         "Cross Greeks drive hedging P&L when spot and vol move together."),
        ("Hedging", "Continuous and frictionless, which is the assumption behind the formula.",
         "Discrete hedging, costs, the gamma against theta trade off, and pin risk near expiry."),
        ("Time", "Calendar days (365) or trading hours (252 days of 6.5 hours) for intraday expiries.",
         "Business day calendars, holidays, weekend theta, event vol."),
        ("Scenario P&L", "Full revaluation with a parallel vol shift, spot and vol shocks independent.",
         "Spot and vol are correlated (vol usually rises when spot falls), and shifts differ by strike."),
    ]
    body = "".join(f"<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>" for a, b, c in rows)
    return ('<div class="tm blot txt"><table><tr><th>Component</th><th>This app</th>'
            f'<th>What a desk will push on</th></tr>{body}</table></div>')


# ------------------------------------------------------------------------------------------------- figures
def _style(fig, height, margin):
    fig.update_layout(
        height=height, paper_bgcolor=C["panel"], plot_bgcolor=C["panel"],
        font=dict(family=FONT, size=12, color=C["ink2"]), margin=margin, showlegend=False,
        hoverlabel=dict(bgcolor=C["panel2"], bordercolor=C["axis"], font=dict(family=FONT, size=12, color=C["ink"])),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=C["axis"], tickfont=dict(size=12, color=C["muted"]))
    fig.update_yaxes(gridcolor=C["line"], gridwidth=1, zeroline=False, linecolor=C["axis"],
                     tickfont=dict(size=12, color=C["muted"]))


def _label(fig, text, idx=1):
    """Panel label at the top left of subplot number idx, kept inside the figure so it never meets a legend."""
    ax = "" if idx == 1 else str(idx)
    fig.add_annotation(xref=f"x{ax} domain", yref=f"y{ax} domain", x=0, y=1.0, text=text, showarrow=False,
                       xanchor="left", yanchor="bottom", font=dict(size=13, color=C["ink2"]))


def payoff_figure(c, d, spots, curves):
    fig = go.Figure()
    labels = {0.0: "Today", 0.5: "Half way to expiry", 1.0: "At expiry"}
    style = {1.0: dict(color=C["muted"], width=1.5, dash="dash"),
             0.5: dict(color=C["time"], width=1.5, dash="dot"),
             0.0: dict(color=C["ink"], width=2.25)}
    for f in (1.0, 0.5, 0.0):
        fig.add_trace(go.Scatter(x=spots, y=curves[f], mode="lines", name=labels[f], line=style[f],
                                 hovertemplate="%{y:+,.2f}<extra>" + labels[f] + "</extra>"))
    fig.add_vline(x=c.K, line_dash="dash", line_color=C["axis"], line_width=1)
    fig.add_vline(x=c.S, line_dash="dot", line_color=C["ink2"], line_width=1)
    fig.add_annotation(xref="x", yref="paper", x=c.K, y=0.0, text=f"K {c.K:,.2f}", showarrow=False,
                       xanchor="left", yanchor="bottom", xshift=5, font=dict(size=11, color=C["muted"]))
    fig.add_annotation(xref="x", yref="paper", x=c.S, y=1.0, text=f"Spot {c.S:,.2f}", showarrow=False,
                       xanchor="left", yanchor="top", xshift=5, font=dict(size=11, color=C["ink2"]))
    be = breakevens(spots, curves[1.0])
    if be:
        fig.add_trace(go.Scatter(x=be, y=[0.0] * len(be), mode="markers", hoverinfo="skip",
                                 marker=dict(symbol="diamond", size=10, color=C["ink"], line=dict(width=2, color=C["panel"]))))
    fig.add_trace(go.Scatter(x=[c.S], y=[0.0], mode="markers", hoverinfo="skip",
                             marker=dict(symbol="circle", size=9, color=C["ink"], line=dict(width=2, color=C["panel"]))))
    _style(fig, 390, dict(l=64, r=20, t=30, b=54))
    fig.update_yaxes(zeroline=True, zerolinecolor=C["axis"], tickformat=",.0f", tickprefix="$")
    fig.update_xaxes(title_text="Spot at the time shown", title_font=dict(size=12, color=C["muted"]), tickformat=",.0f")
    fig.update_layout(hovermode="x unified")
    _label(fig, "Position P&L against spot, from today's price")
    return fig


def payoff_legend():
    return legend_html([("line", C["ink"], "Today"), ("dot", C["time"], "Half way to expiry"),
                        ("dash", C["muted"], "At expiry")])


def greeks_figure(c, spots, curves, fracs):
    colors = {fracs[0]: C["ink"], fracs[1]: C["ramp1"], fracs[2]: C["ramp2"]}
    names = {"delta": "Delta", "gamma": "Gamma", "vega_pt": "Vega per vol point", "theta_day": "Theta per day"}
    fig = make_subplots(rows=2, cols=2, shared_xaxes=False, vertical_spacing=0.14, horizontal_spacing=0.09)
    pos = {"delta": (1, 1), "gamma": (1, 2), "vega_pt": (2, 1), "theta_day": (2, 2)}
    fmt = {"delta": "%{y:+.4f}", "gamma": "%{y:.5f}", "vega_pt": "%{y:.4f}", "theta_day": "%{y:+.4f}"}
    for f in fracs:
        for key, (r, cc) in pos.items():
            fig.add_trace(go.Scatter(x=spots, y=curves[f][key], mode="lines",
                                     line=dict(color=colors[f], width=2.25 if f == fracs[0] else 1.75),
                                     hovertemplate=fmt[key] + "<extra></extra>"), row=r, col=cc)
    for i, key in enumerate(("delta", "gamma", "vega_pt", "theta_day"), start=1):
        _label(fig, names[key], i)
    _style(fig, 650, dict(l=58, r=20, t=34, b=54))
    fig.update_layout(hovermode="x unified")
    fig.add_vline(x=c.K, line_dash="dash", line_color=C["axis"], line_width=1)
    fig.add_vline(x=c.S, line_dash="dot", line_color=C["ink2"], line_width=1)
    fig.update_xaxes(tickformat=",.0f", showspikes=True, spikemode="across", spikesnap="cursor",
                     spikecolor=C["axis"], spikethickness=1)
    fig.update_xaxes(title_text="Spot", title_font=dict(size=12, color=C["muted"]), row=2)
    fig.update_yaxes(tickformat=".2f", row=1, col=1)
    fig.update_yaxes(tickformat=".3f", row=1, col=2)
    fig.update_yaxes(tickformat=".2f", row=2, col=1)
    fig.update_yaxes(tickformat=".3f", row=2, col=2)
    return fig


def greeks_legend(c, fracs):
    def lab(f, tag):
        return f"{tag} ({c.T * f * c.dpy:.1f} d)" if c.dpy != 252.0 else f"{tag} ({c.T * f * 252 * 6.5:.1f} h)"
    return legend_html([("line", C["ink"], lab(fracs[0], "Current expiry")),
                        ("line", C["ramp1"], lab(fracs[1], "Half the time")),
                        ("line", C["ramp2"], lab(fracs[2], "Quarter of the time"))])


def _diverging(lo, hi):
    """Red to neutral to blue with the neutral exactly at zero, each side stretched to its own extreme.

    A long option loses at most its premium but can gain far more, so a symmetric scale would wash out the losses.
    """
    lo, hi = min(lo, -1e-9), max(hi, 1e-9)
    p0 = -lo / (hi - lo)
    if p0 <= 1e-6:
        return [[0.0, C["axis"]], [1.0, C["long"]]], lo, hi
    if p0 >= 1 - 1e-6:
        return [[0.0, C["short"]], [1.0, C["axis"]]], lo, hi
    return [[0.0, C["short"]], [p0, C["axis"]], [1.0, C["long"]]], lo, hi


def scenario_heatmap(c, grid, spot_shocks, vol_shocks, sel=None):
    scale, zlo, zhi = _diverging(float(np.min(grid)), float(np.max(grid)))
    spot_lvl = c.S * (1 + np.asarray(spot_shocks) / 100.0)
    vol_lvl = (c.sigma + np.asarray(vol_shocks) / 100.0) * 100.0
    cd = np.dstack([np.tile(spot_lvl, (len(vol_shocks), 1)), np.tile(vol_lvl[:, None], (1, len(spot_shocks)))])
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=spot_shocks, y=vol_shocks, z=grid, zmin=zlo, zmax=zhi, colorscale=scale, customdata=cd, xgap=0, ygap=0,
        colorbar=dict(thickness=10, len=0.9, outlinewidth=0, tickfont=dict(size=11, color=C["muted"]),
                      tickformat="$,.0f", title=dict(text="P&L", font=dict(size=11, color=C["muted"]))),
        hovertemplate=("Spot %{x:+.1f}% (%{customdata[0]:,.2f})<br>Vol %{y:+.1f} pts (%{customdata[1]:.1f}%)"
                       "<br>P&L %{z:+,.2f}<extra></extra>")))
    fig.add_trace(go.Contour(x=spot_shocks, y=vol_shocks, z=grid, showscale=False, hoverinfo="skip",
                             contours=dict(start=0, end=0, size=1, coloring="none"),
                             line=dict(color=C["ink"], width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=[0.0], y=[0.0], mode="markers", hoverinfo="skip",
                             marker=dict(symbol="cross", size=9, color=C["ink"], line=dict(width=1, color=C["panel"]))))
    if sel is not None:
        fig.add_trace(go.Scatter(x=[sel[0]], y=[sel[1]], mode="markers", hoverinfo="skip",
                                 marker=dict(symbol="diamond", size=12, color="rgba(0,0,0,0)",
                                             line=dict(width=2.5, color=C["ink"]))))
    _style(fig, 540, dict(l=58, r=20, t=30, b=48))
    fig.update_xaxes(title_text="Spot move", ticksuffix="%", tickformat="+.0f", dtick=5,
                     title_font=dict(size=12, color=C["muted"]))
    fig.update_yaxes(title_text="Vol move, points", tickformat="+.0f", dtick=5, gridcolor="rgba(0,0,0,0)",
                     title_font=dict(size=12, color=C["muted"]))
    _label(fig, "Position P&L, full revaluation (dotted line is breakeven, diamond is the scenario below)")
    return fig


def explain_figure(ex):
    names = list(ex["parts"].keys()) + ["Higher order", "Total"]
    vals = list(ex["parts"].values()) + [ex["residual"], ex["full"]]
    fig = go.Figure(go.Waterfall(
        orientation="h", y=names, x=vals, measure=["relative"] * (len(names) - 1) + ["total"],
        text=[usd(v, dec=0) for v in vals], textposition="outside", textfont=dict(size=12, color=C["ink2"]),
        increasing=dict(marker=dict(color=C["long"])), decreasing=dict(marker=dict(color=C["short"])),
        totals=dict(marker=dict(color=C["ink2"])), connector=dict(line=dict(color=C["axis"], width=1)),
        hovertemplate="%{y}: %{x:+,.2f}<extra></extra>", cliponaxis=False))
    cum, pts = 0.0, [0.0]
    for v in vals[:-1]:
        cum += v
        pts.append(cum)
    pts.append(vals[-1])
    lo, hi = min(pts), max(pts)
    span = (hi - lo) or 1.0
    _style(fig, 330, dict(l=110, r=24, t=30, b=40))
    fig.update_yaxes(autorange="reversed", gridcolor="rgba(0,0,0,0)")
    fig.update_xaxes(tickformat="$,.0f", gridcolor=C["line"], showgrid=True, zeroline=True,
                     zerolinecolor=C["axis"], range=[lo - 0.8 * span, hi + 0.25 * span], nticks=3, tickangle=0)
    _label(fig, "P&L by source")
    return fig


def smile_figure(c, ch):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ch["K"], y=ch["iv"] * 100.0, mode="lines+markers",
                             line=dict(color=C["vol"], width=2.25), marker=dict(size=7, color=C["vol"],
                                                                                 line=dict(width=2, color=C["panel"])),
                             hovertemplate="K %{x:,.2f}<br>IV %{y:.2f}%<extra></extra>"))
    fig.add_vline(x=c.S, line_dash="dot", line_color=C["ink2"], line_width=1)
    fig.add_annotation(xref="x", yref="paper", x=c.S, y=0.0, text=f"Spot {c.S:,.2f}", showarrow=False,
                       xanchor="left", yanchor="bottom", xshift=5, font=dict(size=11, color=C["ink2"]))
    _style(fig, 320, dict(l=58, r=20, t=30, b=44))
    fig.update_yaxes(ticksuffix="%", tickformat=".1f")
    fig.update_xaxes(title_text="Strike", title_font=dict(size=12, color=C["muted"]), tickformat=",.0f")
    _label(fig, "Implied vol by strike (illustrative smile)")
    return fig


def iv_figure(c, vols, prices, target, iv=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=vols * 100.0, y=prices, mode="lines", line=dict(color=C["vol"], width=2.25),
                             hovertemplate="Vol %{x:.1f}%<br>Price %{y:,.4f}<extra></extra>"))
    fig.add_hline(y=target, line_dash="dot", line_color=C["ink2"], line_width=1)
    if iv is not None:
        fig.add_trace(go.Scatter(x=[iv * 100.0], y=[target], mode="markers", hoverinfo="skip",
                                 marker=dict(symbol="circle", size=10, color=C["ink"], line=dict(width=2, color=C["panel"]))))
    _style(fig, 320, dict(l=58, r=20, t=30, b=44))
    fig.update_yaxes(tickformat=",.2f", tickprefix="$")
    fig.update_xaxes(title_text="Volatility", ticksuffix="%", title_font=dict(size=12, color=C["muted"]))
    _label(fig, "Option price against vol (strictly increasing, so one root)")
    return fig


def convergence_figure(ns, errs):
    ns = np.asarray(ns, dtype=float)
    errs = np.maximum(np.abs(np.asarray(errs)), 1e-12)
    k = float(np.median(errs * ns))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=errs, mode="lines+markers", line=dict(color=C["vol"], width=2.25),
                             marker=dict(size=7, color=C["vol"], line=dict(width=2, color=C["panel"])),
                             hovertemplate="N = %{x:.0f}<br>|error| %{y:.2e}<extra></extra>"))
    fig.add_trace(go.Scatter(x=ns, y=k / ns, mode="lines", line=dict(color=C["muted"], width=1.25, dash="dash"),
                             hoverinfo="skip"))
    _style(fig, 340, dict(l=64, r=20, t=30, b=44))
    fig.update_xaxes(type="log", title_text="Tree steps N", title_font=dict(size=12, color=C["muted"]))
    fig.update_yaxes(type="log", tickformat=".0e")
    _label(fig, "European tree against Black-Scholes, absolute price error (dashed: 1/N)")
    return fig
