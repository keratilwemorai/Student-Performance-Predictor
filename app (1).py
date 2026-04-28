import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.markdown("""
<style>
    .main { background-color: #f5f0e8; }
    .block-container { padding-top: 0rem; }

    .hero {
        background: #1c1917;
        padding: 48px 40px 36px;
        border-bottom: 4px solid #c84b2f;
        margin: -1rem -1rem 2rem -1rem;
    }
    .hero h1 {
        font-size: 2.8rem;
        color: #f5f0e8;
        margin: 0 0 10px;
        line-height: 1.15;
        font-weight: 700;
    }
    .hero h1 span { color: #c84b2f; }
    .hero p {
        font-family: monospace;
        font-size: 13px;
        color: #a09080;
        max-width: 540px;
        line-height: 1.7;
        margin: 0;
    }

    .result-card {
        background: white;
        border-radius: 12px;
        padding: 22px 18px;
        text-align: center;
        border: 2px solid #e2d9cc;
        margin-bottom: 12px;
    }
    .result-card .label {
        font-family: monospace;
        font-size: 11px;
        color: #78716c;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }
    .result-card .value {
        font-size: 2.2rem;
        font-weight: 900;
        line-height: 1;
        margin-bottom: 6px;
    }
    .result-card .sub {
        font-family: monospace;
        font-size: 11px;
        color: #78716c;
    }

    .tip-box {
        background: #faf7f2;
        border: 1px solid #e2d9cc;
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 8px;
        font-family: monospace;
        font-size: 13px;
        color: #78716c;
        line-height: 1.6;
    }

    .summary-banner {
        background: #1c1917;
        border-radius: 12px;
        padding: 24px 28px;
        color: #c8bdb0;
        font-family: monospace;
        font-size: 13px;
        line-height: 1.8;
        margin-top: 24px;
        border-left: 6px solid #2d7a4f;
    }
    .summary-banner .banner-label {
        font-size: 11px;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #a09080;
        margin-bottom: 10px;
    }

    .stButton > button {
        background: #1c1917 !important;
        color: #f5f0e8 !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 14px !important;
        font-size: 16px !important;
        font-weight: 700 !important;
        width: 100% !important;
        letter-spacing: 0.5px !important;
        box-shadow: 0 4px 24px rgba(28,25,23,0.25) !important;
    }
    .stButton > button:hover {
        background: #c84b2f !important;
    }
</style>
""", unsafe_allow_html=True)

def build_dataset():
    rows = []
    for i in range(120):
        def r(n):
            x = math.sin(i * n + n) * 99999
            return x - math.floor(x)

        attendance    = round(r(3)  * 55 + 45)
        study         = round(r(7)  * 14 + 1)
        prev          = round(r(11) * 45 + 35)
        assignments   = round(r(13) * 9  + 1)
        sleep         = round(r(17) * 5  + 4)
        participation = round(r(19) * 9  + 1)
        internet      = 1 if r(23) > 0.3 else 0
        parent_edu    = round(r(29) * 3)

        score = (
            0.28 * prev +
            0.22 * (attendance * 0.9) +
            0.18 * (study * 6) +
            0.12 * (assignments * 9) +
            0.08 * (participation * 9) +
            0.06 * (min(sleep, 8) * 10) +
            internet * 3 +
            parent_edu * 1.5 +
            (r(31) - 0.5) * 8
        )
        rows.append({
            "attendance":    attendance,
            "study":         study,
            "prev":          prev,
            "assignments":   assignments,
            "sleep":         sleep,
            "participation": participation,
            "internet":      internet,
            "parent_edu":    parent_edu,
            "final":         min(100, max(15, round(score)))
        })
    return pd.DataFrame(rows)

DS = build_dataset()

def predict(att, study, prev, assign, sleep, part, internet, parent_edu):
    weights = [0.28, 0.20, 0.18, 0.12, 0.07, 0.07, 0.04, 0.04]
    norm = [
        att / 100,
        study / 15,
        prev / 100,
        assign / 10,
        min(sleep, 9) / 9,
        part / 10,
        internet,
        parent_edu / 3
    ]
    score = sum(v * w for v, w in zip(norm, weights)) * 100

    def ds_score(row):
        dn = [
            row["attendance"] / 100,
            row["study"] / 15,
            row["prev"] / 100,
            row["assignments"] / 10,
            min(row["sleep"], 9) / 9,
            row["participation"] / 10,
            row["internet"],
            row["parent_edu"] / 3
        ]
        return sum(v * w for v, w in zip(dn, weights)) * 100

    DS["_score"] = DS.apply(ds_score, axis=1)
    pairs = DS[["_score", "final"]].sort_values("_score").reset_index(drop=True)
    lo = pairs[pairs["_score"] >= score].index.min()

    if pd.isna(lo):
        final = int(pairs.iloc[-1]["final"])
    elif lo == 0:
        final = int(pairs.iloc[0]["final"])
    elif lo >= len(pairs):
        final = int(pairs.iloc[-1]["final"])
    else:
        a = pairs.iloc[lo - 1]
        b = pairs.iloc[lo]
        t = (score - a["_score"]) / (b["_score"] - a["_score"] + 0.001)
        final = round(a["final"] + t * (b["final"] - a["final"]))

    final    = min(100, max(12, int(final)))
    grade    = "A" if final >= 80 else "B" if final >= 70 else "C" if final >= 60 else "D" if final >= 50 else "F"
    passfail = "PASS" if final >= 50 else "FAIL"
    risk     = "Low"  if final >= 70 else "Medium" if final >= 55 else "High" if final >= 45 else "Critical"
    return {"final": final, "grade": grade, "passfail": passfail, "risk": risk}

GRADE_COLORS = {"A": "#2d7a4f", "B": "#2563a8", "C": "#b07d2e", "D": "#c07030", "F": "#c84b2f"}
RISK_COLORS  = {"Low": "#2d7a4f", "Medium": "#b07d2e", "High": "#c07030", "Critical": "#c84b2f"}
def grade_color(g): return GRADE_COLORS.get(g, "#78716c")
def risk_color(r):  return RISK_COLORS.get(r, "#78716c")

def get_tips(att, study, prev, assign, sleep, part, internet):
    tips = []
    if att    < 75:    tips.append(("📅", "Attendance below 75% seriously hurts your mark. Aim for 85%+."))
    if study  < 5:     tips.append(("📖", "Try to reach at least 8–10 study hours per week."))
    if sleep  < 6:     tips.append(("😴", "Less than 6 hours sleep reduces memory retention. Aim for 7–8 hrs."))
    if assign < 6:     tips.append(("✏️", "Submit every assignment — even incomplete ones earn partial marks."))
    if part   < 5:     tips.append(("🙋", "Participate more in class. It reinforces learning."))
    if not internet:   tips.append(("📡", "Use school or library Wi-Fi to access study materials."))
    if prev   < 50:    tips.append(("📝", "Revisit foundational topics from previous terms before the exam."))
    if not tips:       tips.append(("🌟", "All indicators look great — keep up these habits!"))
    return tips

def draw_radar(att, study, prev, assign, sleep, part):
    labels = ["Attendance", "Study", "Prev Marks", "Assignments", "Sleep", "Participation"]
    values = [
        att,
        round(study / 15 * 100),
        prev,
        round(assign / 10 * 100),
        round(min(sleep, 9) / 9 * 100),
        round(part / 10 * 100),
    ]
    values += values[:1]
    N      = len(labels)
    angles = [n / float(N) * 2 * math.pi for n in range(N)] + [0]

    fig, ax = plt.subplots(figsize=(4, 4), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#faf7f2")
    ax.plot(angles, values, color="#1c1917", linewidth=2)
    ax.fill(angles, values, color="#1c1917", alpha=0.12)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, size=9, color="#78716c", fontfamily="monospace")
    ax.set_ylim(0, 100)
    ax.set_yticks([25, 50, 75, 100])
    ax.set_yticklabels(["25", "50", "75", "100"], size=7, color="#e2d9cc")
    ax.grid(color="#e2d9cc", linewidth=0.8)
    ax.spines["polar"].set_color("#e2d9cc")
    plt.tight_layout()
    return fig

def draw_contrib(prev, att, study, assign, part, sleep):
    names  = ["Past Marks", "Attendance", "Study Hours", "Assignments", "Participation", "Sleep"]
    vals   = [
        round(prev   * 0.28),
        round(att    * 0.22 * 0.9),
        round(study  * 6    * 0.18),
        round(assign * 9    * 0.12),
        round(part   * 9    * 0.08),
        round(min(sleep, 8) * 10 * 0.06),
    ]
    colors = ["#c84b2f", "#2d7a4f", "#2563a8", "#b07d2e", "#9c4fd8", "#2a9d8f"]

    fig, ax = plt.subplots(figsize=(5, 3.5))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    bars = ax.barh(names, vals, color=colors, height=0.55, edgecolor="none")
    for bar, val in zip(bars, vals):
        ax.text(val + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{val} pts", va="center", ha="left", fontsize=9, color="#78716c", fontfamily="monospace")
    ax.set_xlabel("Contribution (pts)", fontsize=9, color="#78716c", fontfamily="monospace")
    ax.tick_params(axis="y", labelsize=9, colors="#78716c")
    ax.tick_params(axis="x", labelsize=8, colors="#78716c")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#e2d9cc")
    ax.set_xlim(0, max(vals) + 8)
    ax.xaxis.grid(True, color="#e2d9cc", linewidth=0.7, linestyle="--")
    ax.set_axisbelow(True)
    plt.tight_layout()
    return fig

def draw_gauge(value, color):
    fig, ax = plt.subplots(figsize=(4, 2.4))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-0.2, 1.1)
    ax.axis("off")
    theta_bg   = np.linspace(math.pi, 0, 200)
    ax.plot(np.cos(theta_bg), np.sin(theta_bg), color="#e2d9cc", linewidth=18, solid_capstyle="round")
    fill_end   = math.pi - (value / 100) * math.pi
    theta_fill = np.linspace(math.pi, fill_end, 200)
    ax.plot(np.cos(theta_fill), np.sin(theta_fill), color=color, linewidth=18, solid_capstyle="round")
    ax.text(0,     0.28, f"{value}%",      ha="center", va="center", fontsize=28, fontweight="bold", color=color, fontfamily="serif")
    ax.text(0,     0.05, "PREDICTED MARK", ha="center", va="center", fontsize=7.5, color="#78716c", fontfamily="monospace")
    ax.text(-1.05, -0.1, "0%\nFail",       ha="center", fontsize=7, color="#78716c", fontfamily="monospace")
    ax.text(0,     -0.1, "50%\nPass",      ha="center", fontsize=7, color="#78716c", fontfamily="monospace")
    ax.text(1.05,  -0.1, "100%\nDist",     ha="center", fontsize=7, color="#78716c", fontfamily="monospace")
    plt.tight_layout()
    return fig


st.markdown("""
<div class="hero">
    <h1>Student Performance <span>Predictor</span></h1>
    <p>Enter a student's details below to estimate their final mark, grade, and pass/fail outcome.</p>
</div>
""", unsafe_allow_html=True)

col_inputs, col_sidebar = st.columns([2, 1], gap="large")

with col_inputs:
    st.markdown("#### 📚 Academic")
    att    = st.slider("Attendance (%)",              min_value=30, max_value=100, value=75, step=1)
    study  = st.slider("Study Hours per Week (h)",    min_value=0,  max_value=15,  value=7,  step=1)
    prev   = st.slider("Previous Average Mark (%)",   min_value=0,  max_value=100, value=65, step=1)
    assign = st.slider("Assignments Submitted (/10)", min_value=0,  max_value=10,  value=7,  step=1)

    st.markdown("---")
    st.markdown("#### 🧬 Lifestyle & Engagement")
    sleep  = st.slider("Sleep Hours per Night (h)",   min_value=3,  max_value=10,  value=7,  step=1)
    part   = st.slider("Class Participation (/10)",   min_value=1,  max_value=10,  value=6,  step=1)

    st.markdown("**Internet Access**")
    internet_label = st.radio("", ["✅ Yes", "❌ No"], horizontal=True, label_visibility="collapsed")
    internet = 1 if "Yes" in internet_label else 0

    st.markdown("**Parent / Guardian Education**")
    parent_labels = ["🏠 No formal education", "🎓 Matric", "📜 Diploma / Certificate", "🏛️ University Degree"]
    parent_label  = st.radio("", parent_labels, label_visibility="collapsed")
    parent_edu    = parent_labels.index(parent_label)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Performance")

with col_sidebar:
    st.markdown("#### 📡 Student Profile")
    st.pyplot(draw_radar(att, study, prev, assign, sleep, part), use_container_width=True)

    st.markdown("#### 💡 Study Tips")
    tips = get_tips(att, study, prev, assign, sleep, part, internet)
    for icon, tip in tips:
        st.markdown(f'<div class="tip-box">{icon} {tip}</div>', unsafe_allow_html=True)

if predict_btn:
    with st.spinner("Calculating..."):
        result = predict(att, study, prev, assign, sleep, part, internet, parent_edu)

    st.markdown("---")
    st.markdown("<h3 style='text-align:center; color:#1c1917;'>Results</h3>", unsafe_allow_html=True)

    gc = grade_color(result["grade"])
    rc = risk_color(result["risk"])
    pc = "#2d7a4f" if result["passfail"] == "PASS" else "#c84b2f"

    grade_sub = (
        "Excellent"     if result["final"] >= 80 else
        "Good"          if result["final"] >= 70 else
        "Average"       if result["final"] >= 60 else
        "Below average" if result["final"] >= 50 else
        "Failing"
    )
    risk_sub = {
        "Low": "On track", "Medium": "Monitor progress",
        "High": "Needs support", "Critical": "Urgent action needed"
    }[result["risk"]]
    pass_sub = "Will proceed" if result["passfail"] == "PASS" else "Needs support"

    c1, c2, c3, c4 = st.columns(4, gap="small")
    for col, label, value, color, sub in [
        (c1, "Final Mark", f"{result['final']}%", gc, "Predicted score"),
        (c2, "Grade",      result["grade"],        gc, grade_sub),
        (c3, "Status",     result["passfail"],     pc, pass_sub),
        (c4, "Risk",       result["risk"],         rc, risk_sub),
    ]:
        with col:
            st.markdown(f"""
            <div class="result-card">
                <div class="label">{label}</div>
                <div class="value" style="color:{color};">{value}</div>
                <div class="sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    g_col, b_col = st.columns(2, gap="medium")

    with g_col:
        st.markdown("#### 🎯 Performance Meter")
        st.pyplot(draw_gauge(result["final"], gc), use_container_width=True)
        grade_html = ""
        for g, c in [("F", "#c84b2f"), ("D", "#c07030"), ("C", "#b07d2e"), ("B", "#2563a8"), ("A", "#2d7a4f")]:
            if g == result["grade"]:
                grade_html += f'<span style="background:{c};color:white;padding:3px 14px;border-radius:20px;font-family:monospace;font-size:12px;font-weight:700;margin:3px;">{g}</span>'
            else:
                grade_html += f'<span style="background:#ede9e0;color:#78716c;padding:3px 14px;border-radius:20px;font-family:monospace;font-size:12px;margin:3px;">{g}</span>'
        st.markdown(f'<div style="text-align:center;margin-top:8px;">{grade_html}</div>', unsafe_allow_html=True)

    with b_col:
        st.markdown("#### 📊 Factor Breakdown")
        st.pyplot(draw_contrib(prev, att, study, assign, part, sleep), use_container_width=True)

    risk_msg = {
        "Critical": "Immediate academic support is strongly recommended.",
        "High":     "Focused study and improved attendance can shift this outcome.",
        "Medium":   "Consistent effort over the coming weeks can improve this significantly.",
        "Low":      "Keep up these strong habits to secure the result.",
    }[result["risk"]]

    st.markdown(f"""
    <div class="summary-banner" style="border-left-color:{gc};">
        <div class="banner-label">Summary</div>
        With <strong style="color:#f5f0e8;">attendance {att}%</strong>,
        <strong style="color:#f5f0e8;">{study}h/week study</strong>, and a previous average of
        <strong style="color:#f5f0e8;">{prev}%</strong>, the predicted final mark is
        <strong style="color:{gc};">{result['final']}% — Grade {result['grade']}</strong>.
        {risk_msg}
    </div>
    """, unsafe_allow_html=True)
