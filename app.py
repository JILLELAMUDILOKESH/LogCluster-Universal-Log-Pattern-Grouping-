"""Streamlit interface for LogCluster."""

from __future__ import annotations

import html
from io import StringIO

import pandas as pd
import streamlit as st

from logcluster import PatternGroup, discover_patterns, extract_log_text, read_dataset


SAMPLE_LOGS = """2025-03-08 10:42:11 ERROR PaymentService - Timeout connecting to 10.0.1.12
2025-03-08 10:42:13 INFO PaymentService - Retrying request id=8f14e45f
2025-03-08 10:42:14 ERROR PaymentService - Timeout connecting to 10.0.1.15
2025-03-08 10:42:16 INFO PaymentService - Retrying request id=9a71cde2
2025-03-08 10:42:19 WARN AuthService - Login failed for user alice from 192.168.1.42
2025-03-08 10:42:21 WARN AuthService - Login failed for user bob from 192.168.1.55
2025-03-08 10:42:24 INFO WebServer - GET /api/orders/4821 200 34ms
2025-03-08 10:42:25 INFO WebServer - GET /api/orders/7193 200 29ms
2025-03-08 10:42:26 INFO WebServer - GET /api/orders/9284 200 31ms
2025-03-08 10:42:30 ERROR Database - Connection pool exhausted after 30 seconds
2025-03-08 10:42:33 INFO Scheduler - Job cleanup completed in 124ms
2025-03-08 10:42:36 INFO Scheduler - Job cleanup completed in 98ms
2025-03-08 10:42:41 CRITICAL Kernel - Uncorrectable memory error on CPU 2"""
def _make_sample_dataset() -> pd.DataFrame:
   rows = []
   for line in SAMPLE_LOGS.splitlines():
       date, time, message = line.split(" ", 2)
       rows.append({"timestamp": f"{date} {time}", "message": message})
   return pd.DataFrame(rows)


SAMPLE_DATASET = _make_sample_dataset()
SAMPLE_DATASET_CSV = SAMPLE_DATASET.to_csv(index=False).encode("utf-8")

st.set_page_config(
    page_title="LogCluster | Universal Log Pattern Grouping",
    page_icon="📊",
    layout="wide",
)

st.markdown(
    """
    <style>
    :root { --ink: #17213a; --muted: #748099; --blue: #5368df; }
    .stApp {
        background:
            radial-gradient(ellipse at 82% 0%, #e8edff 0, transparent 33%),
            #f5f7fb;
        color: var(--ink);
    }
    .block-container { max-width: 1320px; padding: 1.5rem 3rem 3.5rem; }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fff 0%, #f8f9ff 100%);
        border-right: 1px solid #e7ebf3;
    }
    [data-testid="stSidebar"] section { padding-top: 1.2rem; }
    h1, h2, h3 { font-family: 'Manrope', 'Segoe UI', sans-serif; color: var(--ink); }
    [data-testid="stSidebar"] h2 { font-size: 1rem; margin-top: 1.6rem; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p { color: #72809a; }
    .brand-lockup { display: flex; align-items: center; gap: 11px; margin: 0 0 1.8rem; }
    .brand-mark {
        display: grid; width: 40px; height: 40px; place-items: center;
        border-radius: 12px; background: linear-gradient(145deg, #687bef, #485ac5);
        box-shadow: 0 7px 16px #5368df38; color: #fff;
        font: 800 13px 'Manrope', sans-serif; letter-spacing: -.5px;
    }
    .brand-name { color: #202a42; font: 800 16px 'Manrope', sans-serif; letter-spacing: -.5px; }
    .brand-caption { margin-top: 2px; color: #8b95aa; font-size: 10px; }
    .hero {
        position: relative; min-height: 245px; overflow: hidden; margin: 0 0 24px; padding: 22px 30px 18px;
        border: 1px solid #344577; border-radius: 19px;
        background:
            radial-gradient(circle at 89% 19%, #5369b93d 0, transparent 24%),
            linear-gradient(115deg, #111a31 0%, #1a284a 58%, #263b67 100%);
        box-shadow: 0 18px 38px #1a274017;
        color: #fff;
    }
    .hero::after {
        position: absolute; top: -45px; right: 9%; width: 220px; height: 220px;
        border: 1px solid #ffffff0d; border-radius: 50%; content: "";
        box-shadow: 0 0 0 28px #ffffff08, 0 0 0 58px #ffffff05;
        pointer-events: none;
    }
    .hero-top, .hero-footer, .hero-brand, .hero-status { display: flex; align-items: center; }
    .hero-top, .hero-footer { justify-content: space-between; }
    .hero-brand { gap: 10px; color: #f5f7ff; font: 700 13px 'Manrope', sans-serif; }
    .hero-brand .brand-mark { width: 32px; height: 32px; border: 1px solid #ffffff28; background: #ffffff16; box-shadow: none; }
    .hero-status {
        gap: 8px; padding: 7px 10px; border: 1px solid #ffffff20;
        border-radius: 20px; background: #ffffff0c; color: #d7def1;
        font-size: 9px; font-weight: 700; letter-spacing: .8px;
    }
    .hero-status span { width: 6px; height: 6px; border-radius: 50%; background: #55d5a0; box-shadow: 0 0 0 3px #55d5a022; }
    .hero-copy { position: relative; z-index: 1; max-width: calc(100% - 250px); padding: 15px 0 14px; }
    .hero-eyebrow, .section-kicker {
        color: #a9b8ed; font-size: 9px; font-weight: 700; letter-spacing: 1.45px;
    }
    .hero-title {
        margin-top: 7px; color: #fff; font: 700 clamp(27px, 3vw, 38px)/1.12 'Manrope', sans-serif;
        letter-spacing: -1.35px;
    }
    .hero-title span { color: #aab8ff; }
    .hero-copy p { margin: 11px 0 0; color: #c4cde0; font-size: 12px; }
    .hero-footer { max-width: calc(100% - 250px); padding-top: 14px; border-top: 1px solid #ffffff19; color: #bac5dc; font-size: 10px; }
    .hero-footer strong { color: #eef1fa; font-weight: 600; }
    .hero-visual {
        position: absolute; z-index: 1; top: 50%; right: 30px; width: 218px;
        padding: 14px; transform: translateY(-42%); border: 1px solid #ffffff20;
        border-radius: 13px; background: linear-gradient(145deg, #ffffff16, #ffffff09);
        box-shadow: 0 14px 28px #080f2028; backdrop-filter: blur(8px);
    }
    .visual-heading { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
    .visual-heading span:first-child { color: #c7d1eb; font-size: 8px; font-weight: 700; letter-spacing: 1px; }
    .visual-tag { padding: 4px 6px; border: 1px solid #ffffff20; border-radius: 8px; color: #d3dcf0; font-size: 7px; letter-spacing: .5px; }
    .visual-message { color: #f0f3fb; font: 10px 'Cascadia Code', Consolas, monospace; white-space: nowrap; }
    .visual-message strong { color: #ffb79b; font-weight: 600; }
    .visual-variable { color: #aebeff; }
    .visual-divider { height: 1px; margin: 12px 0 10px; background: #ffffff19; }
    .visual-stat { display: flex; align-items: center; justify-content: space-between; color: #c1cce2; font-size: 8px; }
    .visual-stat strong { color: #fff; font: 700 16px 'Manrope', sans-serif; }
    .visual-bars { display: flex; align-items: end; gap: 4px; height: 24px; margin-top: 11px; }
    .visual-bars span { flex: 1; border-radius: 3px 3px 1px 1px; background: linear-gradient(180deg, #9eafff, #6579e7); opacity: .9; }
    .section-heading { display: flex; align-items: center; gap: 12px; margin: 0 0 15px; }
    .section-number {
        display: grid; width: 30px; height: 30px; place-items: center;
        border: 1px solid #dfe4f1; border-radius: 9px; background: #fff;
        color: #6073da; font-size: 10px; font-weight: 700;
    }
    .section-heading h2 { margin: 0; font: 700 17px 'Manrope', sans-serif; letter-spacing: -.35px; }
    .section-heading p { margin: 3px 0 0; color: #8290a5; font-size: 10px; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: #e6eaf2 !important; border-radius: 14px !important;
        background: #fff; box-shadow: 0 5px 16px #24304e08;
    }
    [data-testid="stMetric"] {
        min-height: 106px; padding: 17px 18px; border: 1px solid #e8ebf2;
        border-radius: 13px; background: #fff; box-shadow: 0 5px 14px #25304c08;
    }
    [data-testid="stMetricLabel"] p { color: #818ba0; font-size: 10px; font-weight: 600; }
    [data-testid="stMetricValue"] { color: #202a42; font: 700 26px 'Manrope', sans-serif; }
    [data-testid="stTabs"] button[role="tab"] { color: #7c879b; font-size: 11px; }
    [data-testid="stTabs"] button[aria-selected="true"] { color: #4d60ce; }
    [data-testid="stTabs"] [data-baseweb="tab-highlight"] { background: #5e72dd; }
    [data-testid="stTextArea"] textarea {
        border-color: #e4e8f0; border-radius: 9px; background: #fbfcfe;
        font-family: 'Cascadia Code', Consolas, monospace; font-size: 11px; line-height: 1.6;
    }
    [data-testid="stFileUploaderDropzone"] {
        border: 1px dashed #cbd3e3; border-radius: 11px; background: #fbfcff;
    }
    [data-testid="stExpander"] {
        border: 1px solid #e6eaf2; border-radius: 10px; background: #fff;
    }
    [data-testid="stRadioOption"] {
        padding: 7px 11px; border: 1px solid transparent; border-radius: 9px;
        transition: background .15s ease, border-color .15s ease;
    }
    [data-testid="stRadioOption"]:hover { background: #f7f8fd; }
    [data-testid="stRadioOption"][data-selected="true"] {
        border-color: #d7ddfb; background: #f0f2ff; color: #4c5fc9;
    }
    [data-testid="stRadioOption"][data-selected="true"] > div > div:first-child > div:first-child {
        border-color: #5b6ed6 !important;
    }
    [data-testid="stRadioOption"][data-selected="true"] > div > div:first-child > div:first-child > div {
        background: #5b6ed6 !important;
    }
    button[kind="primary"] {
        min-height: 42px; border: 0; border-radius: 9px;
        background: linear-gradient(105deg, #596fe2, #485dc9);
        box-shadow: 0 6px 13px #5368df30; font-weight: 700;
    }
    button[kind="primary"]:hover { border: 0; background: linear-gradient(105deg, #4f64d5, #3e52ba); }
    button[kind="secondary"] { border-color: #e4e8f1; border-radius: 8px; color: #596bd2; }
    [data-testid="stDataFrame"] { border: 1px solid #e9edf4; border-radius: 10px; overflow: hidden; }
    [data-testid="stDownloadButton"] button { border-radius: 9px; }
    .pattern-chart {
        margin: 15px 0 7px; padding: 16px 18px 12px;
        border: 1px solid #e8ebf2; border-radius: 13px;
        background: #fff; box-shadow: 0 5px 14px #25304c08;
    }
    .pattern-chart-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
    .pattern-chart-title { color: #34415d; font-size: 11px; font-weight: 700; }
    .pattern-chart-tag { color: #929caf; font-size: 8px; font-weight: 700; letter-spacing: .8px; }
    .pattern-row {
        display: grid; grid-template-columns: minmax(0, 1fr) minmax(90px, 180px) 57px;
        align-items: center; gap: 12px; padding: 7px 0;
    }
    .pattern-label {
        overflow: hidden; color: #59657b; font: 9px 'Cascadia Code', Consolas, monospace;
        text-overflow: ellipsis; white-space: nowrap;
    }
    .pattern-track { height: 6px; overflow: hidden; border-radius: 5px; background: #edf0f7; }
    .pattern-fill { height: 100%; border-radius: inherit; background: linear-gradient(90deg, #7182e8, #5368df); }
    .pattern-count { color: #69758b; font-size: 9px; font-weight: 600; text-align: right; white-space: nowrap; }
    .rare-insight {
        height: 100%; padding: 16px 18px 12px; border: 1px solid #f0e2ca;
        border-radius: 13px; background: linear-gradient(145deg, #fffdfa, #fff8eb);
        box-shadow: 0 5px 14px #25304c08;
    }
    .rare-insight-head { display: flex; align-items: center; gap: 10px; margin-bottom: 12px; }
    .rare-insight-icon {
        display: grid; width: 30px; height: 30px; flex: 0 0 30px; place-items: center;
        border-radius: 9px; background: #fff0d7; color: #ad741d; font-size: 15px; font-weight: 700;
    }
    .rare-insight-title { color: #4a5368; font-size: 11px; font-weight: 700; }
    .rare-insight-subtitle { margin-top: 3px; color: #8d887d; font-size: 9px; line-height: 1.4; }
    .rare-insight-badge {
        margin-left: auto; padding: 4px 7px; border: 1px solid #efdfc3;
        border-radius: 10px; color: #9a6b22; font-size: 8px; font-weight: 700; white-space: nowrap;
    }
    .rare-entry { display: flex; gap: 8px; padding: 9px 0; border-top: 1px solid #f1e8da; }
    .rare-entry-marker { color: #c58a31; font-size: 12px; line-height: 1.2; }
    .rare-template {
        overflow-wrap: anywhere; color: #536078;
        font: 9px/1.45 'Cascadia Code', Consolas, monospace;
    }
    .rare-example {
        overflow-wrap: anywhere; margin-top: 3px; color: #8d96a6;
        font: 8px/1.4 'Cascadia Code', Consolas, monospace;
    }
    .rare-insight-more { margin-top: 7px; color: #9a6b22; font-size: 9px; font-weight: 600; }
    .ready-note { min-height: 20px; color: #78869d; font-size: 10px; }
    .ready-note strong { color: #388365; font-weight: 600; }
    .sidebar-note {
        margin-top: 23px; padding: 13px; border: 1px solid #e5e9f2;
        border-radius: 11px; background: #fff;
    }
    .sidebar-note strong { color: #34425e; font-size: 11px; }
    .sidebar-note p { margin: 5px 0 0; color: #8490a4; font-size: 10px; line-height: 1.5; }
    .method-note { margin-top: 12px; color: #77839a; font-size: 10px; }
    .footer-note { padding-top: 17px; border-top: 1px solid #e7ebf2; color: #929caf; font-size: 9px; }
    @media (max-width: 760px) {
        .block-container { padding: 1.1rem 1rem 2rem; }
        .hero { padding: 20px; border-radius: 15px; }
        .hero-title { font-size: 30px; }
        .hero-copy, .hero-footer { max-width: none; }
        .hero-visual { display: none; }
        .pattern-row { grid-template-columns: minmax(0, 1fr) 74px 48px; gap: 8px; }
        .hero-footer { align-items: flex-start; gap: 8px; flex-direction: column; }
    }
    @media (min-width: 761px) and (max-width: 1040px) {
        .hero-visual { right: 20px; width: 190px; }
        .hero-copy, .hero-footer { max-width: calc(100% - 215px); }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <div class="hero-top">
        <div class="hero-brand"><span class="brand-mark">LC</span><span>LogCluster</span></div>
        <div class="hero-status"><span></span> PRIVATE &amp; LOCAL</div>
      </div>
      <div class="hero-copy">
        <div class="hero-eyebrow">UNSUPERVISED LOG INTELLIGENCE</div>
        <div class="hero-title">Find the <span>signal</span> in your logs.</div>
        <p>Discover recurring patterns and surface unusual events — no labels or setup required.</p>
      </div>
      <div class="hero-visual" aria-hidden="true">
        <div class="visual-heading"><span>EXAMPLE CLUSTER</span><span class="visual-tag">AUTO-GROUPED</span></div>
        <div class="visual-message"><strong>ERROR</strong> api.request failed</div>
        <div class="visual-message">host=<span class="visual-variable">&lt;IP&gt;</span></div>
        <div class="visual-divider"></div>
        <div class="visual-stat"><span>similar events</span><strong>12</strong></div>
        <div class="visual-bars"><span style="height:42%"></span><span style="height:68%"></span><span style="height:48%"></span><span style="height:88%"></span><span style="height:62%"></span><span style="height:100%"></span><span style="height:76%"></span><span style="height:91%"></span></div>
      </div>
      <div class="hero-footer">
        <span>✦ &nbsp;<strong>Automatic pattern discovery</strong></span>
        <span>🔒 &nbsp;Your logs stay in this app session</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        """
        <div class="brand-lockup">
          <div class="brand-mark">LC</div>
          <div><div class="brand-name">LogCluster</div><div class="brand-caption">Universal log explorer</div></div>
        </div>
        <div class="section-kicker">MODEL CONTROLS</div>
        """,
        unsafe_allow_html=True,
    )
    st.header("Clustering settings")
    similarity = st.slider(
        "Similarity threshold",
        min_value=0.35,
        max_value=0.95,
        value=0.68,
        step=0.01,
        help="Higher values make clusters more strict. Lower values group more varied messages together.",
    )
    min_samples = st.slider(
        "Minimum events per pattern",
        min_value=2,
        max_value=10,
        value=2,
        help="A pattern needs at least this many events to form a cluster. Unclustered lines are shown as rare events.",
    )
    st.caption("Tune how similar messages need to be before they group.")
    with st.expander("How does clustering work?"):
        st.write(
            "LogCluster generalizes common variable values, then compares message "
            "shapes and groups similar lines automatically. No labeled examples "
            "or predefined pattern list is needed."
        )
    st.markdown(
        """
        <div class="sidebar-note">
          <strong>◈ &nbsp;Private by design</strong>
          <p>Logs are analyzed in this app session and are never sent to an external analysis service.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="section-heading">
      <div class="section-number">01</div>
      <div><h2>Bring your logs</h2><p>Paste log lines, load a sample, or upload a dataset.</p></div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.container(border=True):
    source = st.radio(
        "Choose input",
        ["Paste log lines", "Upload a file"],
        horizontal=True,
        label_visibility="collapsed",
    )
    log_text = ""
    if source == "Paste log lines":
        input_label, sample_action = st.columns([3, 1])
        with input_label:
            st.caption("LOG CONTENT · ONE EVENT PER LINE")
        with sample_action:
            if st.button("✦ Load sample logs", type="secondary"):
                st.session_state["log_input"] = SAMPLE_LOGS
                st.rerun()
        log_text = st.text_area(
            "Log input",
            height=190,
            placeholder="Paste log lines here…\n\n2025-03-08 10:42:11 ERROR PaymentService - Timeout connecting to 10.0.1.12",
            label_visibility="collapsed",
            key="log_input",
        )
    else:
        uploaded_file = st.file_uploader(
            "Drop a dataset or log file here",
            type=["csv", "json", "jsonl", "log", "txt"],
            help="For a dataset, choose the column that contains the log messages.",
        )
        sample_info, sample_download = st.columns([3, 1])
        with sample_info:
            st.caption("For structured datasets, keep one event per row and select its message column.")
        with sample_download:
            st.download_button(
                "Download sample CSV",
                data=SAMPLE_DATASET_CSV,
                file_name="logcluster-sample.csv",
                mime="text/csv",
            )
        if uploaded_file is not None:
            file_content = uploaded_file.getvalue()
            extension = uploaded_file.name.rsplit(".", 1)[-1].casefold()
            if extension in {"log", "txt"}:
                log_text = file_content.decode("utf-8-sig", errors="replace")
                st.caption(f"Log file · {uploaded_file.name}")
            else:
                try:
                    dataset = read_dataset(uploaded_file.name, file_content)
                except (ValueError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
                    st.error(f"Could not read this dataset. Check the file format and encoding. {error}")
                else:
                    if dataset.empty or len(dataset.columns) == 0:
                        st.error("This dataset has no rows or columns to analyze.")
                    else:
                        column_names = [str(column) for column in dataset.columns]
                        preferred_names = {
                            "message", "messages", "log", "logs", "log message",
                            "log_message", "event", "text", "content", "details",
                        }
                        default_index = next(
                            (index for index, name in enumerate(column_names) if name.casefold() in preferred_names),
                            None,
                        )
                        if default_index is None:
                            text_columns = {
                                str(column)
                                for column in dataset.select_dtypes(include=["object", "string"]).columns
                            }
                            default_index = next(
                                (index for index, name in enumerate(column_names) if name in text_columns),
                                0,
                            )
                        selected_name = st.selectbox(
                            "Column containing log messages",
                            column_names,
                            index=default_index,
                            help="Choose the field whose values are log messages or event text.",
                        )
                        selected_column = dataset.columns[column_names.index(selected_name)]
                        log_text = extract_log_text(dataset, selected_column)
                        st.caption(
                            f"Dataset · {len(dataset):,} rows · {len(dataset.columns)} columns · "
                            f"Analyzing “{selected_name}”"
                        )
                        if not log_text:
                            st.warning("The selected column has no non-empty values. Choose another column.")

    if source == "Upload a file" and log_text.strip():
        with st.expander("Preview the first 3 log events"):
            preview = [line for line in log_text.splitlines() if line.strip()][:3]
            st.code("\n".join(preview), language=None)

    action_note, action_button = st.columns([3, 1])
    with action_note:
        ready_lines = sum(bool(line.strip()) for line in log_text.splitlines())
        if ready_lines:
            st.markdown(f'<div class="ready-note">✓ &nbsp;<strong>{ready_lines:,} events ready</strong> &nbsp;·&nbsp; Values like timestamps and IPs are generalized.</div>', unsafe_allow_html=True)
        elif source == "Upload a file":
            st.markdown('<div class="ready-note">Choose a dataset or log file to get started.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="ready-note">Add log lines or load the sample to get started.</div>', unsafe_allow_html=True)
    with action_button:
        analyze_clicked = st.button(
            "✳ Discover patterns",
            type="primary",
            disabled=not log_text.strip(),
        )

if analyze_clicked:
    try:
        line_count, patterns = discover_patterns(
            log_text,
            similarity=similarity,
            min_samples=min_samples,
        )
        st.session_state["analysis"] = {
            "line_count": line_count,
            "patterns": patterns,
        }
    except ValueError as error:
        st.error(str(error))

analysis = st.session_state.get("analysis")
if analysis:
    patterns: list[PatternGroup] = analysis["patterns"]
    line_count: int = analysis["line_count"]
    rare_count = sum(pattern.count == 1 for pattern in patterns)
    repeated_count = sum(pattern.count > 1 for pattern in patterns)

    st.markdown(
        """
        <div class="section-heading">
          <div class="section-number">02</div>
          <div><h2>Your log landscape</h2><p>Patterns are grouped by message similarity, without predefined labels.</p></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metrics = st.columns(4)
    metrics[0].metric("Log lines", f"{line_count:,}")
    metrics[1].metric("Patterns found", f"{len(patterns):,}")
    metrics[2].metric("Rare events", f"{rare_count:,}", help="Patterns seen only once.")
    metrics[3].metric("Repeated events", f"{repeated_count:,}", help="Patterns seen more than once.")

    top_repeated = [pattern for pattern in patterns if pattern.count > 1][:5]
    chart_html = ""
    if top_repeated:
        max_count = top_repeated[0].count
        chart_rows = "".join(
            f'<div class="pattern-row">'
            f'<div class="pattern-label" title="{html.escape(pattern.template, quote=True)}">{html.escape(pattern.template)}</div>'
            f'<div class="pattern-track"><div class="pattern-fill" style="width:{pattern.count / max_count * 100:.2f}%"></div></div>'
            f'<div class="pattern-count">{pattern.count:,} events</div>'
            f'</div>'
            for pattern in top_repeated
        )
        chart_html = (
            '<div class="pattern-chart"><div class="pattern-chart-head">'
            '<span class="pattern-chart-title">Most repeated patterns</span>'
            '<span class="pattern-chart-tag">TOP EVENTS</span></div>'
            f'{chart_rows}</div>'
        )

    rare_patterns = [pattern for pattern in patterns if pattern.count == 1]
    rare_html = ""
    if rare_patterns:
        rare_rows = "".join(
            '<div class="rare-entry"><span class="rare-entry-marker">!</span>'
            '<div><div class="rare-template">'
            f'{html.escape(pattern.template)}'
            '</div><div class="rare-example">'
            f'{html.escape(pattern.example)}'
            '</div></div></div>'
            for pattern in rare_patterns[:3]
        )
        more_count = len(rare_patterns) - 3
        more_note = (
            f'<div class="rare-insight-more">+ {more_count:,} more — see the Rare events tab below.</div>'
            if more_count > 0 else ""
        )
        rare_label = f'{len(rare_patterns):,} ONE-OFF' + ("S" if len(rare_patterns) != 1 else "")
        rare_html = (
            '<div class="rare-insight" role="note">'
            '<div class="rare-insight-head"><span class="rare-insight-icon">!</span>'
            '<div><div class="rare-insight-title">Worth a closer look</div>'
            '<div class="rare-insight-subtitle">Appeared once · review context before acting</div></div>'
            f'<span class="rare-insight-badge">{rare_label}</span></div>'
            f'{rare_rows}{more_note}</div>'
        )

    if chart_html and rare_html:
        chart_column, rare_column = st.columns([1.2, 1], gap="medium")
        with chart_column:
            st.markdown(chart_html, unsafe_allow_html=True)
        with rare_column:
            st.markdown(rare_html, unsafe_allow_html=True)
    elif chart_html:
        st.markdown(chart_html, unsafe_allow_html=True)
    elif rare_html:
        st.markdown(rare_html, unsafe_allow_html=True)

    all_tab, rare_tab = st.tabs([
        f"✳  All patterns · {len(patterns)}",
        f"◈  Rare events · {rare_count}",
    ])
    for tab, rare_only in ((all_tab, False), (rare_tab, True)):
        with tab:
            with st.container(border=True):
                query = st.text_input(
                    "Search patterns",
                    placeholder="Search patterns or example log lines...",
                    key="rare_search" if rare_only else "all_search",
                ).casefold()
                selected = [
                    pattern for pattern in patterns
                    if (not rare_only or pattern.count == 1)
                    and (not query or query in pattern.template.casefold() or query in pattern.example.casefold())
                ]
                table = pd.DataFrame([
                    {
                        "Pattern": pattern.template,
                        "Example": pattern.example,
                        "Count": pattern.count,
                        "Share": round(pattern.count / line_count * 100, 1),
                        "Status": "Rare" if pattern.count == 1 else "Group",
                    }
                    for pattern in selected
                ])
                if table.empty:
                    st.info("No patterns match this view.")
                else:
                    st.dataframe(
                        table,
                        column_config={
                            "Pattern": st.column_config.TextColumn("Pattern", width=250),
                            "Example": st.column_config.TextColumn("Example", width=200),
                            "Count": st.column_config.NumberColumn("Count", width=70),
                            "Share": st.column_config.ProgressColumn(
                                "Share",
                                help="Percentage of all input log lines matching this pattern.",
                                min_value=0,
                                max_value=100,
                                format="%.1f%%",
                                width=100,
                                color="blue",
                            ),
                            "Status": st.column_config.TextColumn("Type", width=75),
                        },
                        width="stretch",
                        hide_index=True,
                    )
                st.caption(f"Showing {len(selected):,} of {len(patterns):,} patterns.")

    csv_buffer = StringIO()
    pd.DataFrame([
        {
            "Pattern": pattern.template,
            "Occurrences": pattern.count,
            "Share (%)": round(pattern.count / line_count * 100, 2),
            "Example": pattern.example,
        }
        for pattern in patterns
    ]).to_csv(csv_buffer, index=False)
    st.download_button(
        "↓ Export results as CSV",
        data=csv_buffer.getvalue(),
        file_name="logcluster-patterns.csv",
        mime="text/csv",
    )
    st.markdown(
        '<p class="method-note">✦ &nbsp;<strong>HOW IT WORKS</strong> &nbsp; '
        "How discovery works: timestamps, IP addresses, IDs, and numbers are generalized; "
        'DBSCAN groups similar messages without pre-labeled examples. One-off groups are marked rare.</p>',
        unsafe_allow_html=True,
    )

st.markdown(
    '<div class="footer-note">LOGCLUSTER &nbsp;·&nbsp; UNIVERSAL LOG PATTERN GROUPING '
    '&nbsp;·&nbsp; UNSUPERVISED ANALYSIS</div>',
    unsafe_allow_html=True,
)
