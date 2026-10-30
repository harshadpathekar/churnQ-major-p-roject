"""ChurnQ: one-click Excel report for a single scored file.

One workbook, three sheets, plain-English labels:
  Dashboard           key numbers, risk levels, segments, with charts
  Retention Strategy  what to do for each customer group
  Customer List       every customer, highest risk first, colour-coded
"""
import io
import json
import os
import datetime

import pandas as pd
from openpyxl import Workbook
from openpyxl.chart import BarChart, Reference
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

INK, ORANGE, GREEN, AMBER, GRAY = "1F2226", "E4502B", "A9CF8E", "F2B84B", "D9DBDE"
WHITE = "FFFFFF"
RISK_FILL = {"High": "F4B9A8", "Medium": "F8E0A8", "Low": "D4EBC3"}


# ---------------------------------------------------------------- loading
def _find(folder, filename, suffix):
    for name in (f"predict_{filename}_{suffix}", f"{filename}_{suffix}"):
        p = os.path.join(folder, name)
        if os.path.exists(p):
            return p
    return None


def _read_csv(path):
    return pd.read_csv(path) if path else None


def _read_json(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _shared_unique_key(a, b):
    for c in a.columns:
        if c in b.columns and a[c].is_unique and b[c].is_unique:
            return c
    return None


def _attach(base, other, allow_position=False):
    """Add columns of `other` that `base` lacks, matched on a shared unique ID column.
    Position matching is used only when allowed and the lengths agree."""
    if other is None:
        return base
    new_cols = [c for c in other.columns if c not in base.columns]
    if not new_cols:
        return base
    key = _shared_unique_key(base, other)
    if key:
        return base.merge(other[[key] + new_cols], on=key, how="left")
    if allow_position and len(base) == len(other):
        out = base.copy()
        for c in new_cols:
            out[c] = other[c].values
        return out
    return base


def _col(df, *names):
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None


# ---------------------------------------------------------------- numbers
def _pct_scale(series):
    p = pd.to_numeric(series, errors="coerce")
    return p * (100 if p.max() <= 1 else 1)


def _summary(df):
    s = {"customers": int(len(df))}
    pc = _col(df, "predicted_churn")
    is_churn = (df[pc].astype(str).str.lower().isin(["yes", "1", "true"])
                if pc else pd.Series(False, index=df.index))
    s["predicted_churners"] = int(is_churn.sum())
    s["churn_rate"] = round(100 * is_churn.mean(), 1) if len(df) else 0.0
    rc = _col(df, "risk_level")
    s["risk"] = {k: int((df[rc] == k).sum()) for k in ("High", "Medium", "Low")} if rc else {}
    pb = _col(df, "churn_probability")
    s["avg_probability"] = round(float(_pct_scale(df[pb]).mean()), 1) if pb else None
    mc = _col(df, "MonthlyCharges")
    s["revenue_at_risk"] = (round(float(pd.to_numeric(df.loc[is_churn, mc], errors="coerce").sum()), 2)
                            if mc else None)
    return s, is_churn


def _segment_table(df, is_churn):
    sc = _col(df, "segment", "customer_segment") or next(
        (c for c in df.columns if "segment" in c.lower()), None)
    if not sc:
        return None
    rc, mc = _col(df, "risk_level"), _col(df, "MonthlyCharges")
    rows = []
    for name, g in df.groupby(sc):
        ch = is_churn.loc[g.index]
        row = {"Customer group": name, "Customers": len(g),
               "Likely to leave (%)": round(100 * ch.mean(), 1)}
        if rc:
            row["High risk (%)"] = round(100 * (g[rc] == "High").mean(), 1)
        if mc:
            m = pd.to_numeric(g[mc], errors="coerce")
            row["Average monthly bill"] = round(m.mean(), 2)
            row["Monthly revenue at risk"] = round(float(m[ch].sum()), 2)
        rows.append(row)
    out = pd.DataFrame(rows)
    sort_col = "Monthly revenue at risk" if "Monthly revenue at risk" in out else "Customers"
    return out.sort_values(sort_col, ascending=False).reset_index(drop=True)


def _retention_items(ret):
    if ret is None:
        return []
    if isinstance(ret, dict):
        for k in ("segments", "strategies", "results"):
            if isinstance(ret.get(k), (list, dict)):
                ret = ret[k]
                break
    items = []
    if isinstance(ret, dict):
        for name, v in ret.items():
            if isinstance(v, dict):
                items.append({"segment": name, **v})
    elif isinstance(ret, list):
        items = [v for v in ret if isinstance(v, dict)]
    for it in items:
        if not it.get("segment"):
            it["segment"] = it.get("name") or it.get("segment_name") or it.get("title") or ""
    return items


def _actions_list(v):
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return [str(x) for x in v]
    return [str(v)]


# ---------------------------------------------------------------- excel helpers
def _fill(hex_):
    return PatternFill("solid", start_color=hex_, end_color=hex_)


def _header(ws, row, labels, col=1):
    for i, lab in enumerate(labels):
        c = ws.cell(row=row, column=col + i, value=lab)
        c.font = Font(bold=True, color=WHITE)
        c.fill = _fill(INK)
        c.alignment = Alignment(horizontal="left", vertical="center", wrap_text=True)


def _section(ws, row, text):
    c = ws.cell(row=row, column=1, value=text)
    c.font = Font(bold=True, size=13, color=ORANGE)


def _widths(ws, widths):
    for i, w in enumerate(widths, 1):
        ws.column_dimensions[get_column_letter(i)].width = w


def _bar_chart(title, ws, cats_ref, vals_ref, color, anchor, y_title):
    ch = BarChart()
    ch.type = "col"
    ch.title = title
    ch.legend = None
    ch.y_axis.title = y_title
    ch.add_data(vals_ref, titles_from_data=True)
    ch.set_categories(cats_ref)
    ch.series[0].graphicalProperties.solidFill = color
    ch.y_axis.scaling.min = 0
    ch.height, ch.width = 6.2, 13
    ch.y_axis.delete = False
    ch.x_axis.delete = False
    ws.add_chart(ch, anchor)


# ---------------------------------------------------------------- sheets
def _sheet_dashboard(wb, filename, s, seg):
    ws = wb.active
    ws.title = "Dashboard"
    ws.sheet_view.showGridLines = False
    _widths(ws, [30, 16, 20, 18, 22, 26])

    ws.merge_cells("A1:F1")
    t = ws["A1"]
    t.value = "ChurnQ: Customer Churn Report"
    t.font = Font(bold=True, size=18, color=WHITE)
    t.fill = _fill(INK)
    t.alignment = Alignment(vertical="center", indent=1)
    ws.row_dimensions[1].height = 34
    ws["A2"] = f"File: {filename}    |    Created: {datetime.datetime.now():%d %b %Y, %H:%M}"
    ws["A2"].font = Font(italic=True, color="666666")

    _section(ws, 4, "At a glance")
    glance = [
        ("Customers checked", s["customers"]),
        ("Customers likely to leave", s["predicted_churners"]),
        ("Share likely to leave", f"{s['churn_rate']}%"),
        ("High-risk customers", s.get("risk", {}).get("High", "n/a")),
        ("Average chance of leaving",
         f"{s['avg_probability']}%" if s.get("avg_probability") is not None else "n/a"),
        ("Monthly revenue at risk", s["revenue_at_risk"] if s.get("revenue_at_risk") is not None else "n/a"),
    ]
    for i, (lab, val) in enumerate(glance):
        r = 5 + i
        a = ws.cell(row=r, column=1, value=lab)
        a.fill = _fill("F1F1EF")
        b = ws.cell(row=r, column=2, value=val)
        b.font = Font(bold=True, size=12)
        b.alignment = Alignment(horizontal="left")
        b.fill = _fill("F1F1EF")
        if isinstance(val, (int, float)):
            b.number_format = "#,##0.00" if isinstance(val, float) else "#,##0"

    nxt = 5 + len(glance) + 1
    risk = s.get("risk") or {}
    if risk:
        _section(ws, nxt, "Customers by risk level")
        _header(ws, nxt + 1, ["Risk level", "Customers", "Share of all"])
        for i, k in enumerate(("High", "Medium", "Low")):
            r = nxt + 2 + i
            cell = ws.cell(row=r, column=1, value=k)
            cell.fill = _fill(RISK_FILL[k])
            ws.cell(row=r, column=2, value=risk.get(k, 0)).number_format = "#,##0"
            share = ws.cell(row=r, column=3,
                            value=(risk.get(k, 0) / s["customers"]) if s["customers"] else 0)
            share.number_format = "0.0%"
        first, last = nxt + 2, nxt + 4
        _bar_chart("Customers by risk level", ws,
                   Reference(ws, min_col=1, min_row=first, max_row=last),
                   Reference(ws, min_col=2, min_row=nxt + 1, max_row=last),
                   ORANGE, "D4", "Customers")
        nxt = last + 2

    if seg is not None and not seg.empty:
        _section(ws, nxt, "Customer groups (most revenue at risk first)")
        _header(ws, nxt + 1, list(seg.columns))
        for i, row in enumerate(seg.itertuples(index=False)):
            for j, v in enumerate(row):
                c = ws.cell(row=nxt + 2 + i, column=1 + j, value=v)
                if j > 0 and isinstance(v, float):
                    c.number_format = "#,##0.00"
        first, last = nxt + 2, nxt + 1 + len(seg)
        val_col = (list(seg.columns).index("Monthly revenue at risk") + 1
                   if "Monthly revenue at risk" in seg.columns else 2)
        title = ("Monthly revenue at risk by group" if val_col != 2 else "Customers by group")
        _bar_chart(title, ws,
                   Reference(ws, min_col=1, min_row=first, max_row=last),
                   Reference(ws, min_col=val_col, min_row=nxt + 1, max_row=last),
                   INK, f"A{last + 2}", "Revenue" if val_col != 2 else "Customers")
        nxt = last + 2 + 15

    _section(ws, nxt, "How to read this report")
    notes = [
        "A customer is marked 'likely to leave' when the system's chance of leaving is 55% or more.",
        "Risk level: High = 66% or more, Medium = 33% to 66%, Low = below 33%.",
        "This is a screening tool: it helps you decide who to contact first. It is not a certainty about any one customer.",
        "See the 'Retention Strategy' sheet for what to do, and 'Customer List' for every customer.",
    ]
    for i, n in enumerate(notes):
        ws.merge_cells(start_row=nxt + 1 + i, start_column=1, end_row=nxt + 1 + i, end_column=6)
        c = ws.cell(row=nxt + 1 + i, column=1, value=n)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        ws.row_dimensions[nxt + 1 + i].height = 30


def _sheet_retention(wb, items):
    ws = wb.create_sheet("Retention Strategy")
    ws.sheet_view.showGridLines = False
    _widths(ws, [5, 26, 12, 20, 48, 72])
    ws.merge_cells("A1:F1")
    t = ws["A1"]
    t.value = "What to do for each customer group"
    t.font = Font(bold=True, size=16, color=WHITE)
    t.fill = _fill(INK)
    t.alignment = Alignment(vertical="center", indent=1)
    ws.row_dimensions[1].height = 30
    if not items:
        ws["A3"] = ("Retention strategies have not been generated for this file yet. "
                    "Open the Retention Strategy page in the app, then download again.")
        return
    _header(ws, 3, ["#", "Customer group", "Priority", "Best way to reach them",
                    "What is happening", "What to do"])
    pri_fill = {"high": RISK_FILL["High"], "medium": RISK_FILL["Medium"], "low": RISK_FILL["Low"]}
    for i, it in enumerate(items):
        r = 4 + i
        actions = _actions_list(it.get("actions"))
        action_text = "\n".join(f"{n}. {a}" for n, a in enumerate(actions, 1))
        diag = str(it.get("diagnosis", "") or "")
        vals = [i + 1, it.get("segment", ""), str(it.get("priority", "") or "").title(),
                it.get("channel", ""), diag, action_text]
        for j, v in enumerate(vals):
            c = ws.cell(row=r, column=1 + j, value=v)
            c.alignment = Alignment(wrap_text=True, vertical="top")
        pf = pri_fill.get(str(it.get("priority", "")).lower())
        if pf:
            ws.cell(row=r, column=3).fill = _fill(pf)
        ws.cell(row=r, column=2).font = Font(bold=True)
        lines = max(len(actions) + 1, len(diag) // 45 + 1, 3)
        ws.row_dimensions[r].height = min(15 * lines, 400)
    ws.freeze_panes = "A4"


def _sheet_customers(wb, customers):
    ws = wb.create_sheet("Customer List")
    df = customers.copy()
    pb, rc = _col(df, "churn_probability"), _col(df, "risk_level")
    pc, sc = _col(df, "predicted_churn"), _col(df, "segment", "customer_segment")
    idc = _col(df, "customerID", "customer_id", "id")
    if pb:
        df[pb] = _pct_scale(df[pb]).round(1)
        df = df.sort_values(pb, ascending=False)
    rename = {}
    if rc: rename[rc] = "Risk level"
    if pb: rename[pb] = "Chance of leaving (%)"
    if pc: rename[pc] = "Likely to leave?"
    if sc: rename[sc] = "Customer group"
    first = [c for c in (idc, rc, pb, pc, sc) if c]
    rest = [c for c in df.columns if c not in first]
    df = df[first + rest].rename(columns=rename)
    fl = df.select_dtypes("float").columns
    df[fl] = df[fl].round(2)

    _header(ws, 1, list(df.columns))
    risk_idx = list(df.columns).index("Risk level") + 1 if rc else None
    for r, row in enumerate(df.itertuples(index=False), start=2):
        for c, v in enumerate(row, start=1):
            if pd.isna(v):
                v = None
            ws.cell(row=r, column=c, value=v)
        if risk_idx:
            val = ws.cell(row=r, column=risk_idx).value
            if val in RISK_FILL:
                ws.cell(row=r, column=risk_idx).fill = _fill(RISK_FILL[val])
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = ws.dimensions
    for i, col in enumerate(df.columns, 1):
        sample = df[col].astype(str).head(200)
        ws.column_dimensions[get_column_letter(i)].width = min(
            max(len(str(col)), int(sample.str.len().max() or 0)) + 3, 32)


# ---------------------------------------------------------------- public
def build_report_xlsx(processed_folder, filename, df=None, retention=None):
    """Return (xlsx_bytes, download_name).

    df         scored customers with a 'segment' column (the app passes the output of
               _load_segmented). When omitted, the saved files in processed_folder are used.
    retention  retention strategies (list or dict). When omitted, the saved
               <file>_retention.json is used if it exists.
    Raises FileNotFoundError if no predictions can be found."""
    if df is not None:
        customers = df.copy()
        ret = retention if retention is not None else _read_json(
            _find(processed_folder, filename, "retention.json"))
    else:
        p_pred = _find(processed_folder, filename, "predictions.csv")
        if not p_pred:
            raise FileNotFoundError(
                "No predictions found for this file yet. Run the prediction step first.")
        pred = _read_csv(p_pred)
        seg = _read_csv(_find(processed_folder, filename, "segments.csv"))
        orig = _read_csv(_find(processed_folder, filename, "original.csv"))
        ret = retention if retention is not None else _read_json(
            _find(processed_folder, filename, "retention.json"))
        customers = _attach(pred, seg, allow_position=True)
        customers = _attach(customers, orig, allow_position=False)

    s, is_churn = _summary(customers)
    seg_tbl = _segment_table(customers, is_churn)
    items = _retention_items(ret)

    wb = Workbook()
    _sheet_dashboard(wb, filename, s, seg_tbl)
    _sheet_retention(wb, items)
    _sheet_customers(wb, customers)

    buf = io.BytesIO()
    wb.save(buf)
    stem = os.path.splitext(filename)[0].replace(" ", "_")
    return buf.getvalue(), f"ChurnQ_Report_{stem}.xlsx"