"""
SupportPilot — AI Agent Co-pilot for E-commerce Customer Support
Streamlit front-end.  Run locally with:  streamlit run app.py
"""
import os
import time

import numpy as np
import pandas as pd
import streamlit as st

import engine as E

st.set_page_config(page_title="SupportPilot · AI Support Co-pilot", page_icon="🎧", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 1.6rem;}
.hero {background: linear-gradient(120deg,#0f766e 0%,#115e59 55%,#134e4a 100%); color:#fff;
       padding: 1.6rem 1.8rem; border-radius: 14px; margin-bottom: 1rem;}
.hero h1 {color:#fff; margin:0 0 .3rem 0; font-size:2.1rem;}
.hero p {color:#d1fae5; margin:0; font-size:1.05rem;}
.pill {display:inline-block; padding:2px 10px; border-radius:999px; font-size:.8rem; font-weight:600;
       margin-right:6px;}
.p1 {background:#fee2e2; color:#991b1b;} .p2 {background:#fef3c7; color:#92400e;} .p3 {background:#dcfce7; color:#166534;}
.small {font-size:.85rem; color:#6b7280;}
</style>
""", unsafe_allow_html=True)


# ============================================================================ cached resources
@st.cache_resource(show_spinner="Training intent model on 1,200 labelled tickets…")
def get_pilot():
    data = E.load_data()
    model, report = E.train_intent_model(data["tickets"])
    return E.SupportPilot(data, model), data, report


pilot, data, report = get_pilot()
tickets = data["tickets"]
AVG_AHT = float(tickets["baseline_handle_min"].mean())
AHT_BY_INTENT = tickets.groupby("intent")["baseline_handle_min"].mean().to_dict()


@st.cache_data(show_spinner="Triaging the open ticket queue…")
def triage_queue(n: int = 150):
    sample = tickets.sort_values("created_at", ascending=False).head(n)
    rows, t0 = [], time.perf_counter()
    for _, t in sample.iterrows():
        a = pilot.analyse(t["message"], t["customer_id"])
        rows.append({
            "ticket_id": t["ticket_id"], "channel": t["channel"], "created": t["created_at"],
            "priority": a.priority, "score": a.priority_score, "intent": a.intent_label,
            "confidence": round(a.confidence, 2), "sentiment": a.sentiment, "tier": (a.customer or {}).get("tier", "-"),
            "recommended_action": a.action, "escalate": a.escalate, "message": t["message"],
        })
    elapsed = time.perf_counter() - t0
    return pd.DataFrame(rows).sort_values("score", ascending=False), elapsed


def secret(name: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(name, os.getenv(name, default)))
    except Exception:
        return os.getenv(name, default)


# ============================================================================ sidebar
with st.sidebar:
    st.markdown("## 🎧 SupportPilot")
    st.caption("AI co-pilot for e-commerce support agents")
    page = st.radio("Navigate", ["🏠 Overview", "🎧 Agent Workspace", "📥 Smart Ticket Queue",
                                 "💰 Business Impact", "📊 Model Evaluation", "🏗️ How It Works"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown("**Reply generator**")
    providers = {
        "Offline (grounded templates)": None,
        "Groq (free tier · Llama 3.3 70B)": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", "GROQ_API_KEY"),
        "OpenAI (GPT-4o mini)": (None, "gpt-4o-mini", "OPENAI_API_KEY"),
    }
    default_provider = 1 if secret("GROQ_API_KEY") else (2 if secret("OPENAI_API_KEY") else 0)
    provider = st.selectbox("Mode", list(providers), index=default_provider, label_visibility="collapsed")
    llm_cfg = None
    if providers[provider]:
        base_url, model_name, key_name = providers[provider]
        key = secret(key_name) or st.text_input(f"{key_name}", type="password",
                                                help="Stored only for this session. Or set it in Streamlit secrets.")
        model_name = st.text_input("Model", value=secret("LLM_MODEL", model_name))
        llm_cfg = {"api_key": key, "base_url": base_url, "model": model_name}
        if not key:
            st.info("No key yet — replies will use offline templates.")
    st.divider()
    st.caption("Demo data is synthetic (fictional retailer *NovaMart*). Inspired by Wayfair's Agent Co-pilot "
               "case study in the ZenML LLMOps Database.")


# ============================================================================ helpers
def pill(priority: str) -> str:
    cls = "p1" if priority.startswith("P1") else ("p2" if priority.startswith("P2") else "p3")
    return f'<span class="pill {cls}">{priority}</span>'


def money(x: float, cur: str = "₹") -> str:
    if cur == "₹":
        if abs(x) >= 1e7:
            return f"₹{x / 1e7:,.2f} Cr"
        if abs(x) >= 1e5:
            return f"₹{x / 1e5:,.2f} L"
        return f"₹{x:,.0f}"
    return f"${x:,.0f}"


if "sent" not in st.session_state:
    st.session_state.sent = []
if "msg" not in st.session_state:
    st.session_state.msg = ""


# ============================================================================ pages
def page_overview():
    st.markdown("""<div class="hero"><h1>SupportPilot</h1>
    <p>An AI co-pilot that reads every customer ticket, pulls the order and policy, recommends the right action
    and drafts the reply — so human agents resolve tickets faster, more consistently, and at lower cost.</p></div>""",
                unsafe_allow_html=True)

    imp = E.impact(2000, round(AVG_AHT, 1), 0.25, 35000)
    c = st.columns(4)
    c[0].metric("Handle time per ticket", f"{imp['aht_after']:.1f} min", f"-{AVG_AHT - imp['aht_after']:.1f} min vs manual",
                delta_color="inverse")
    c[1].metric("Agent capacity freed", f"{imp['fte_freed']:.1f} FTE", "at 2,000 tickets/day")
    c[2].metric("Net annual saving", money(imp["net_annual_saving"]), "after AI running cost")
    c[3].metric("Intent accuracy (unseen msgs)", f"{report['holdout_accuracy'] * 100:.0f}%",
                f"{report['test_accuracy'] * 100:.0f}% on test split", delta_color="off")
    st.caption("Expected scenario: 25% handle-time reduction · ₹35,000/month fully-loaded agent cost · details on "
               "the Business Impact page.")

    l, r = st.columns([1.1, 1])
    with l:
        st.subheader("The business problem")
        st.markdown(f"""
- Support agents spend most of each ticket **searching** — order system, courier status, policy docs — before they type a word.
- Average manual handle time in our data: **{AVG_AHT:.1f} minutes per ticket**; peaks during sales (Diwali, Big Billion-style events) create backlogs.
- Replies are **inconsistent**: different agents quote different refund timelines and compensation, causing repeat contacts and escalations.
- Angry, high-value and Gold-tier customers wait in the same FIFO queue as simple "where is my order?" questions.
""")
        st.subheader("The solution")
        st.markdown("""
For every ticket SupportPilot does in **< 50 ms** what an agent does manually in minutes:
1. **Understands** the request — intent, sentiment, urgency
2. **Looks up** the order, customer tier and product
3. **Retrieves** the exact company policy that applies (RAG)
4. **Recommends** the next best action — refund, replace, escalate…
5. **Drafts** a grounded, on-brand reply the agent can edit and send

The agent stays in control (**human-in-the-loop**) — the AI never sends anything on its own.
""")
    with r:
        st.subheader("Real-world reference case")
        with st.container(border=True):
            st.markdown("**Wayfair — AI-Powered Co-pilot for Digital Sales Agents** (ZenML LLMOps Database)")
            st.markdown("Wayfair built an LLM co-pilot that suggests chat responses to its agents using product "
                        "information, company policies and conversation history. Initial tests showed about a "
                        "**10% reduction in handle time**.")
            st.markdown("<span class='small'>We use 10% as our *conservative* scenario and model 25% / 40% for "
                        "expected / optimistic cases, because SupportPilot also automates lookup, triage and "
                        "action selection.</span>", unsafe_allow_html=True)
        st.subheader("Try it in 30 seconds")
        st.markdown("Open **🎧 Agent Workspace** → pick a ticket → click **Analyse** → edit and **Approve & send**.")
        st.subheader("Dataset used")
        st.markdown(f"""
| Table | Rows |
|---|---|
| Labelled support tickets | {len(tickets):,} |
| Orders | {len(data['orders']):,} |
| Customers | {len(data['customers']):,} |
| Products | {len(data['products']):,} |
| Policy articles (knowledge base) | {len(data['policies'])} |
""")


def page_workspace():
    st.title("🎧 Agent Workspace")
    st.caption("This is what a support agent sees. Pick a real ticket from the queue or type a new one.")

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        src = st.radio("Ticket source", ["Pick from queue", "Write my own"], horizontal=True)
        if src == "Pick from queue":
            sample = tickets.sample(40, random_state=5)
            options = {f"{r.ticket_id} · {r.channel} · {r.message[:70]}…": r for r in sample.itertuples()}
            choice = st.selectbox("Open ticket", list(options))
            t = options[choice]
            message, cust_id = t.message, t.customer_id
            st.text_area("Customer message", message, height=110, disabled=True)
        else:
            message = st.text_area("Customer message", height=130,
                                   placeholder="e.g. My headphones ORD-50012 arrived broken!! Need a replacement asap",
                                   value="I received my Noise Cancelling Headphones (ORD-50358) but the left side "
                                         "has no sound and the box was crushed. Really disappointed!")
            cust_id = None
            st.caption("Tip: include an order ID between ORD-50000 and ORD-50599 so SupportPilot can look it up.")

        go = st.button("⚡ Analyse with SupportPilot", type="primary", use_container_width=True)
        if go and message.strip():
            t0 = time.perf_counter()
            a = pilot.analyse(message, cust_id)
            a = pilot.draft_reply(a, llm_cfg)
            st.session_state.analysis = a
            st.session_state.latency = (time.perf_counter() - t0) * 1000
            st.session_state.run_id = st.session_state.get("run_id", 0) + 1

        a = st.session_state.get("analysis")
        if a is not None:
            st.markdown("#### Context pulled automatically")
            if a.order:
                o = a.order
                with st.container(border=True):
                    st.markdown(f"**Order {o['order_id']}** · {o['status']}")
                    oc = st.columns(2)
                    oc[0].markdown(f"🛍️ {o['product_name']} × {o['quantity']}<br>💳 ₹{o['amount']:,} · "
                                   f"{o['payment_method']}<br>🗓️ Ordered {o['order_date']}", unsafe_allow_html=True)
                    oc[1].markdown(f"🚚 {o['courier']} · {o['tracking_id']}<br>📦 Promised {o['promised_delivery']}"
                                   f"<br>✅ Delivered {o['delivered_on'] or '—'}", unsafe_allow_html=True)
            else:
                st.info("No order ID detected in this message.")
            if a.customer:
                cu = a.customer
                with st.container(border=True):
                    st.markdown(f"**{cu['name']}** · {cu['tier']} member · {cu['city']} · "
                                f"{cu['lifetime_orders']} lifetime orders")
            with st.expander(f"📚 Policies retrieved ({len(a.policies)})", expanded=False):
                for p in a.policies:
                    st.markdown(f"**{p['id']} · {p['title']}**  \n{p['text']}")

    with right:
        a = st.session_state.get("analysis")
        if a is None:
            st.info("👈 Choose a ticket and click **Analyse** to see the co-pilot in action.")
            return
        m = st.columns([1.6, 1])
        m[0].metric("Intent", a.intent_label)
        m[1].metric("Confidence", f"{a.confidence * 100:.0f}%")
        m2 = st.columns([1.6, 1])
        m2[0].metric("Sentiment", a.sentiment)
        m2[1].metric("Priority score", a.priority_score)
        st.markdown(pill(a.priority) + " " + " ".join(
            f"<span class='pill' style='background:#e0e7ff;color:#3730a3'>{r}</span>" for r in a.priority_reasons),
                    unsafe_allow_html=True)
        st.write("")
        if a.escalate:
            st.error(f"**Recommended action: {a.action}** — {a.action_detail}  \n⚠️ Escalate to Tier-2 supervisor.")
        else:
            st.success(f"**Recommended action: {a.action}** — {a.action_detail}")
        for f in a.flags:
            st.warning(f)

        st.markdown(f"#### ✍️ Draft reply  <span class='small'>· {a.reply_source} · generated in "
                    f"{st.session_state.get('latency', 0):.0f} ms</span>", unsafe_allow_html=True)
        # A fresh widget key per analysis, seeded with the draft, survives page switches reliably.
        reply = st.text_area("Edit before sending", value=a.reply, key=f"reply_{st.session_state.get('run_id', 0)}",
                             height=260, label_visibility="collapsed")

        manual = AHT_BY_INTENT.get(a.intent, AVG_AHT)
        assisted = manual * 0.75
        c = st.columns(3)
        c[0].metric("Manual handle time", f"{manual:.1f} min")
        c[1].metric("With co-pilot (est.)", f"{assisted:.1f} min", f"-{manual - assisted:.1f} min", delta_color="inverse")
        if c[2].button("✅ Approve & send", use_container_width=True):
            st.session_state.sent.append({
                "intent": a.intent_label, "action": a.action, "edited": reply.strip() != a.reply.strip(),
                "minutes_saved": manual - assisted,
            })
            st.toast("Reply sent to customer (simulated).", icon="📨")

        if st.session_state.sent:
            log = pd.DataFrame(st.session_state.sent)
            st.divider()
            s = st.columns(3)
            s[0].metric("Tickets handled this session", len(log))
            s[1].metric("Agent minutes saved", f"{log.minutes_saved.sum():.1f}")
            s[2].metric("Drafts sent without edits", f"{(~log.edited).mean() * 100:.0f}%")


def page_queue():
    st.title("📥 Smart Ticket Queue")
    st.caption("Instead of first-in-first-out, SupportPilot triages every incoming ticket and puts the urgent ones on top.")
    q, elapsed = triage_queue(150)
    c = st.columns(5)
    c[0].metric("Tickets triaged", len(q))
    c[1].metric("Time taken by AI", f"{elapsed:.2f} s")
    c[2].metric("Manual triage (≈1 min each)", f"{len(q)} min")
    c[3].metric("P1 urgent", int(q.priority.str.startswith("P1").sum()))
    c[4].metric("Need escalation", int(q.escalate.sum()))

    f = st.columns(3)
    pri = f[0].multiselect("Priority", sorted(q.priority.unique()), default=sorted(q.priority.unique()))
    intents = f[1].multiselect("Intent", sorted(q.intent.unique()))
    sentiment = f[2].multiselect("Sentiment", sorted(q.sentiment.unique()))
    view = q[q.priority.isin(pri)]
    if intents:
        view = view[view.intent.isin(intents)]
    if sentiment:
        view = view[view.sentiment.isin(sentiment)]
    st.dataframe(view.drop(columns=["score"]), use_container_width=True, hide_index=True, height=420,
                 column_config={"confidence": st.column_config.ProgressColumn("confidence", min_value=0, max_value=1),
                                "message": st.column_config.TextColumn("message", width="large")})
    g = st.columns(2)
    with g[0]:
        st.markdown("**What customers are contacting us about**")
        st.bar_chart(q.intent.value_counts(), horizontal=True, color="#0f766e")
    with g[1]:
        st.markdown("**Priority mix**")
        st.bar_chart(q.priority.value_counts().sort_index(), color="#f59e0b")
    st.info("💡 Insight for managers: the intent mix tells you where to fix the root cause — e.g. a spike in "
            "'Where is my order' means proactive delivery notifications could prevent those tickets entirely.")


def page_impact():
    st.title("💰 Business Impact")
    st.caption("How SupportPilot saves money. Every input is an assumption you can change live during the demo.")

    cur = st.radio("Currency", ["₹ INR", "$ USD"], horizontal=True)
    usd = cur.startswith("$")
    fx = 88.0  # assumption: ₹88 = $1
    sym = "$" if usd else "₹"

    with st.container(border=True):
        st.markdown("**Assumptions** (mid-size Indian e-commerce company)")
        a = st.columns(3)
        tpd = a[0].number_input("Tickets per day", 100, 50000, 2000, step=100)
        aht = a[1].number_input("Manual handle time (min/ticket)", 2.0, 30.0, round(AVG_AHT, 1), step=0.5,
                                help="Default = average in our labelled ticket data.")
        cost = a[2].number_input(f"Agent cost per month ({sym}, fully loaded)", 100.0, 500000.0,
                                 400.0 if usd else 35000.0, step=50.0 if usd else 1000.0,
                                 help="Salary + PF + seat, laptop, supervision overhead.")
        b = st.columns(3)
        red = b[0].slider("Handle-time reduction with co-pilot", 0.05, 0.60, 0.25, 0.01, format="%.2f",
                          help="Wayfair reported ~10% in initial tests; 25% is our expected case.")
        llm = b[1].number_input(f"AI cost per ticket ({sym})", 0.0, 50.0, 0.005 if usd else 0.40, step=0.05,
                                format="%.3f", help="≈1,500 tokens per ticket on a low-cost LLM + hosting.")
        util = b[2].slider("Agent utilisation", 0.5, 0.95, 0.75, 0.05,
                           help="Share of paid time actually spent on tickets.")

    imp = E.impact(tpd, aht, red, cost, llm_cost_per_ticket=llm, utilisation=util)
    fmt = (lambda x: f"${x:,.0f}") if usd else money
    k = st.columns(4)
    k[0].metric("Agent hours saved / month", f"{imp['hours_saved']:,.0f} h")
    k[1].metric("Capacity freed", f"{imp['fte_freed']:.1f} agents", f"of {imp['fte_before']:.1f} needed today",
                delta_color="off")
    k[2].metric("Net saving / month", fmt(imp["net_monthly_saving"]))
    k[3].metric("Net saving / year", fmt(imp["net_annual_saving"]))
    k2 = st.columns(4)
    k2[0].metric("Cost per ticket", f"{sym}{imp['cost_per_ticket_after']:.2f}",
                 f"{sym}{imp['cost_per_ticket_after'] - imp['cost_per_ticket_before']:.2f}", delta_color="inverse")
    k2[1].metric("Handle time", f"{imp['aht_after']:.1f} min", f"-{aht - imp['aht_after']:.1f} min", delta_color="inverse")
    k2[2].metric("AI running cost / month", fmt(imp["llm_cost"]))
    k2[3].metric("Return on AI spend", f"{imp['roi_multiple']:.0f}×" if np.isfinite(imp["roi_multiple"]) else "∞",
                 help="Labour saved ÷ AI running cost")

    c = st.columns(2)
    with c[0]:
        st.markdown("**Monthly support cost: before vs after**")
        chart = pd.DataFrame({"Agent labour": [imp["labour_before"], imp["labour_after"]],
                              "AI running cost": [0, imp["llm_cost"]]}, index=["Today (manual)", "With SupportPilot"])
        st.bar_chart(chart, stack=True, color=["#0f766e", "#f59e0b"])
    with c[1]:
        st.markdown("**Annual net saving vs ticket volume**")
        vols = np.linspace(250, max(tpd * 2, 1000), 12).astype(int)
        curve = pd.DataFrame({
            "Conservative (10%)": [E.impact(v, aht, 0.10, cost, llm_cost_per_ticket=llm, utilisation=util)["net_annual_saving"] for v in vols],
            "Expected (25%)": [E.impact(v, aht, 0.25, cost, llm_cost_per_ticket=llm, utilisation=util)["net_annual_saving"] for v in vols],
            "Optimistic (40%)": [E.impact(v, aht, 0.40, cost, llm_cost_per_ticket=llm, utilisation=util)["net_annual_saving"] for v in vols],
        }, index=pd.Index(vols, name="tickets/day"))
        st.line_chart(curve, color=["#94a3b8", "#0f766e", "#22c55e"])

    st.markdown("#### Scenario summary")
    rows = []
    for name, r in [("Conservative (Wayfair's reported 10%)", 0.10), ("Expected", 0.25), ("Optimistic", 0.40)]:
        x = E.impact(tpd, aht, r, cost, llm_cost_per_ticket=llm, utilisation=util)
        rows.append({"Scenario": name, "AHT reduction": f"{r:.0%}", "AHT after (min)": round(x["aht_after"], 1),
                     "Agents freed": round(x["fte_freed"], 1), "Hours saved / month": round(x["hours_saved"]),
                     "Net saving / year": fmt(x["net_annual_saving"])})
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    st.markdown("#### Success criteria — how we prove it worked")
    st.dataframe(pd.DataFrame([
        ["Average handle time (AHT)", f"{aht:.1f} min", f"≤ {aht * 0.75:.1f} min (-25%)", "Helpdesk timestamps, A/B: co-pilot vs control agents"],
        ["Cost per ticket", f"{sym}{imp['cost_per_ticket_before']:.2f}", f"{sym}{imp['cost_per_ticket_after']:.2f}", "Monthly support cost ÷ tickets"],
        ["Draft acceptance rate", "—", "≥ 60% sent with no / minor edits", "Compare draft vs sent text"],
        ["Intent accuracy", "—", "≥ 90% on unseen messages", "Weekly sample reviewed by QA team"],
        ["First response time (P1)", "Same as P3 (FIFO)", "< 15 min", "Queue timestamps by priority"],
        ["Repeat contact rate", "baseline", "-15%", "Same customer + order within 7 days"],
        ["CSAT", "baseline", "+5 pts", "Post-chat survey"],
        ["Policy compliance", "manual QA", "100% replies cite correct policy", "Automated check on retrieved policy"],
    ], columns=["KPI", "Today", "Target", "How we measure"]), hide_index=True, use_container_width=True)
    with st.expander("Formula used"):
        st.markdown(r"""
- Monthly tickets = tickets/day × 26 working days
- Agent hours needed = tickets × AHT ÷ 60 ÷ utilisation
- Agents (FTE) needed = hours ÷ (8 h × 26 days)
- Labour cost = FTE × monthly agent cost
- **Net saving = labour before − labour after − AI running cost**

Freed capacity can be realised as lower hiring for festive peaks, re-deploying agents to sales/retention, or
absorbing growth without new hires — not necessarily lay-offs.""")
    if usd:
        st.caption(f"Currency conversion assumption: ₹{fx:.0f} = $1.")


def page_eval():
    st.title("📊 Model Evaluation")
    st.caption("Can we trust the AI? We measure it on held-out data — and on messages written in completely new words.")
    c = st.columns(4)
    c[0].metric("Accuracy · test split", f"{report['test_accuracy'] * 100:.1f}%", f"{report['test_size']} tickets",
                delta_color="off")
    c[1].metric("Accuracy · unseen phrasing", f"{report['holdout_accuracy'] * 100:.1f}%",
                f"{report['holdout_size']} hand-written msgs", delta_color="off")
    t0 = time.perf_counter()
    for msg in tickets.message.head(200):
        pilot.analyse(msg)
    ms = (time.perf_counter() - t0) / 200 * 1000
    c[2].metric("Analysis latency", f"{ms:.1f} ms / ticket")
    c[3].metric("Intents covered", len(E.INTENT_LABELS))
    st.info("Why two numbers? The test split comes from the same synthetic generator as training data, so it is "
            "optimistic. The 'unseen phrasing' set uses wording the model never saw — that's the honest number we "
            "would track in production.")

    st.markdown("#### Per-intent performance (test split)")
    pc = report["per_class"].copy()
    pc.index = [E.INTENT_LABELS[i] for i in pc.index]
    st.dataframe(pc[["precision", "recall", "f1-score", "support"]].round(3), use_container_width=True)

    st.markdown("#### Confusion matrix (test split)")
    try:
        st.dataframe(report["confusion"].style.background_gradient(cmap="Greens"), use_container_width=True)
    except Exception:
        st.dataframe(report["confusion"], use_container_width=True)

    st.markdown("#### Unseen-phrasing test")
    st.dataframe(report["holdout_table"], hide_index=True, use_container_width=True)

    st.markdown("#### Guardrails built in")
    st.markdown("""
- **Human-in-the-loop:** the agent reviews and edits every draft before sending.
- **Grounded replies:** the LLM only sees the order record and retrieved policy and is told not to invent facts.
- **Low-confidence flag:** if intent confidence < 50%, the agent is warned to verify.
- **Security rule:** replies never ask for card number, CVV or OTP.
- **Fail-safe:** if the LLM API fails, the app falls back to grounded templates — the demo never breaks.
""")


def page_how():
    st.title("🏗️ How It Works")
    st.graphviz_chart("""
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#ecfdf5", color="#0f766e", fontname="Helvetica", fontsize=11];
  edge [color="#64748b"];
  T [label="Customer ticket\\n(Email / Chat / WhatsApp / App)", fillcolor="#fef3c7", color="#b45309"];
  I [label="1. Intent classifier\\nTF-IDF + Logistic Regression"];
  S [label="2. Sentiment & priority\\nrules: anger, tier, value, repeat"];
  X [label="3. Entity lookup\\norder · customer · product"];
  R [label="4. Policy retrieval (RAG)\\nTF-IDF over knowledge base"];
  A [label="5. Next-best-action\\nbusiness rules"];
  L [label="6. Reply drafting\\nLLM (Groq/OpenAI) or template"];
  H [label="Human agent\\nreviews · edits · sends", fillcolor="#e0e7ff", color="#3730a3"];
  D [label="Orders DB", shape=cylinder, fillcolor="#f1f5f9"];
  K [label="Policy KB", shape=cylinder, fillcolor="#f1f5f9"];
  T -> I; T -> S; T -> X; X -> D [dir=both, style=dashed]; R -> K [dir=both, style=dashed];
  I -> R; I -> A; X -> A; S -> A; R -> L; A -> L; L -> H;
}""")
    c = st.columns(2)
    with c[0]:
        st.markdown("#### Tech stack")
        st.markdown("""
| Layer | Tool |
|---|---|
| Front-end & hosting | Streamlit · Streamlit Community Cloud |
| ML | scikit-learn (TF-IDF, Logistic Regression) |
| Retrieval | TF-IDF vector search over policy KB |
| LLM (optional) | Llama 3.3 70B on Groq, or GPT-4o mini |
| Data | pandas · synthetic dataset (1,200 tickets) |
| Built with | AI coding agent (Claude) |
""")
    with c[1]:
        st.markdown("#### From demo to production")
        st.markdown("""
- Connect to the real helpdesk (Freshdesk / Zendesk / Salesforce) and order management system via APIs
- Replace TF-IDF with an embedding model + vector DB as the knowledge base grows
- Retrain the intent model weekly on agent-corrected labels (feedback loop)
- Run an A/B test: 50% agents with co-pilot vs 50% without, for 4 weeks, to measure AHT and CSAT
- Monitor: draft acceptance rate, hallucination reports, latency, cost per ticket
""")
    st.markdown("#### Limitations (being honest)")
    st.markdown("Synthetic data is cleaner than real tickets; real accuracy would be measured after a pilot. "
                "Impact numbers are assumption-based estimates, not observed results. Sentiment uses rules, which "
                "can miss sarcasm. Multi-issue tickets get a single primary intent.")


{"🏠 Overview": page_overview, "🎧 Agent Workspace": page_workspace, "📥 Smart Ticket Queue": page_queue,
 "💰 Business Impact": page_impact, "📊 Model Evaluation": page_eval, "🏗️ How It Works": page_how}[page]()
