"""
SupportPilot engine — the "brain" of the co-pilot, kept separate from the UI.

Pipeline for every incoming ticket:
  1. Intent classification   (TF-IDF + Logistic Regression, trained on labelled tickets)
  2. Sentiment & priority     (lexicon rules + business rules: tier, order value, repeat contact)
  3. Entity extraction        (order ID -> order, customer and product records)
  4. Policy retrieval (RAG)   (TF-IDF search over the policy knowledge base)
  5. Next-best-action         (business rules: refund / replace / escalate ...)
  6. Reply drafting           (LLM if an API key is configured, otherwise a grounded template)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline

DATA = Path(__file__).parent / "data"
TODAY = date(2026, 10, 14)  # "today" inside the synthetic world
ORDER_RE = re.compile(r"\bORD-?\s?(\d{5})\b", re.IGNORECASE)

INTENT_LABELS = {
    "where_is_my_order": "Where is my order",
    "delivery_delay": "Delivery delayed",
    "damaged_item": "Damaged / defective item",
    "wrong_item": "Wrong item delivered",
    "return_request": "Return / exchange",
    "refund_status": "Refund status",
    "cancel_order": "Cancel order",
    "payment_issue": "Payment issue",
    "product_question": "Product question",
    "account_issue": "Account / login",
}

NEG_WORDS = {"ridiculous", "disappointed", "worst", "angry", "upset", "pathetic", "terrible", "fraud",
             "cheated", "useless", "horrible", "unacceptable", "fix this", "never again", "scam",
             "broken", "damaged", "defective", "wrong", "twice", "double", "still", "not here",
             "second time", "third time", "no update"}
STRONG_NEG = {"fraud", "cheated", "scam", "consumer forum", "legal", "court", "worst", "pathetic",
              "never again", "unacceptable", "ridiculous"}
POS_WORDS = {"thanks", "thank you", "great", "love", "appreciate", "happy", "awesome"}
REPEAT_WORDS = {"second time", "third time", "again", "already contacted", "no reply", "still waiting"}

# Unseen, hand-written test messages (different phrasing from the training templates).
# Used to report an honest "real-world" accuracy, not just accuracy on look-alike data.
HOLDOUT = [
    ("yo my parcel hasn't shown up, any idea where it is?", "where_is_my_order"),
    ("Could you check the shipping status of my purchase", "where_is_my_order"),
    ("is my package out for delivery today", "where_is_my_order"),
    ("what's the ETA on my shipment", "where_is_my_order"),
    ("the courier said delivery by monday, it's thursday now, still nothing!!", "delivery_delay"),
    ("promised date long gone, why so late", "delivery_delay"),
    ("delivery got postponed 3 times, this is unacceptable", "delivery_delay"),
    ("screen is cracked on arrival", "damaged_item"),
    ("the blender doesn't work at all, it's dead out of the box", "damaged_item"),
    ("item came with a big dent and missing parts", "damaged_item"),
    ("this isn't my product, you shipped a different model", "wrong_item"),
    ("received a blue one, I had ordered black", "wrong_item"),
    ("the package had some random item instead of my shoes", "wrong_item"),
    ("shoes are too tight, I want to send them back", "return_request"),
    ("how can I exchange this for a bigger size", "return_request"),
    ("not satisfied with quality, want to give it back", "return_request"),
    ("it's been 10 days since pickup and no money back", "refund_status"),
    ("refund not received in my bank yet", "refund_status"),
    ("when does the amount get credited back after return", "refund_status"),
    ("I don't want it anymore, stop the order please", "cancel_order"),
    ("ordered by mistake, please don't ship it", "cancel_order"),
    ("need to cancel what I bought this morning", "cancel_order"),
    ("my upi got debited but app says payment failed", "payment_issue"),
    ("I've been billed two times for the same purchase", "payment_issue"),
    ("card charged but no confirmation email received", "payment_issue"),
    ("does this have bluetooth 5.3?", "product_question"),
    ("what's the warranty period on the headphones", "product_question"),
    ("is this item water resistant", "product_question"),
    ("not getting the otp to sign in", "account_issue"),
    ("forgot my password and the reset link doesn't work", "account_issue"),
    ("how do I change the email on my profile", "account_issue"),
]


# ============================================================================ data
def load_data() -> dict:
    return {
        "products": pd.read_csv(DATA / "products.csv"),
        "customers": pd.read_csv(DATA / "customers.csv"),
        "orders": pd.read_csv(DATA / "orders.csv", keep_default_na=False),
        "tickets": pd.read_csv(DATA / "tickets.csv", keep_default_na=False),
        "policies": json.loads((DATA / "policies.json").read_text()),
    }


def normalise(text: str) -> str:
    text = ORDER_RE.sub(" ordernumber ", str(text))
    text = re.sub(r"\b(rs\.?|inr|₹)\s?\d[\d,]*", " moneyamount ", text, flags=re.IGNORECASE)
    return text.lower()


# ============================================================================ intent model
def build_intent_model() -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(preprocessor=normalise, ngram_range=(1, 2), min_df=1, sublinear_tf=True)),
        ("char", TfidfVectorizer(preprocessor=normalise, analyzer="char_wb", ngram_range=(3, 5),
                                 sublinear_tf=True)),
    ])
    return Pipeline([("features", features), ("clf", LogisticRegression(max_iter=2000, C=4.0))])


def train_intent_model(tickets: pd.DataFrame, seed: int = 7):
    """Train on 80% of tickets; return the fitted model plus an evaluation report."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        tickets["message"], tickets["intent"], test_size=0.2, random_state=seed, stratify=tickets["intent"])
    model = build_intent_model().fit(X_tr, y_tr)

    pred = model.predict(X_te)
    labels = list(INTENT_LABELS)
    hold_x = [m for m, _ in HOLDOUT]
    hold_y = [y for _, y in HOLDOUT]
    hold_pred = model.predict(hold_x)

    report = {
        "test_size": len(X_te),
        "test_accuracy": accuracy_score(y_te, pred),
        "holdout_size": len(HOLDOUT),
        "holdout_accuracy": accuracy_score(hold_y, hold_pred),
        "per_class": pd.DataFrame(classification_report(
            y_te, pred, labels=labels, output_dict=True, zero_division=0)).T.loc[labels],
        "confusion": pd.DataFrame(confusion_matrix(y_te, pred, labels=labels),
                                  index=[INTENT_LABELS[l] for l in labels],
                                  columns=[INTENT_LABELS[l] for l in labels]),
        "holdout_table": pd.DataFrame({
            "message": hold_x,
            "true_intent": [INTENT_LABELS[y] for y in hold_y],
            "predicted": [INTENT_LABELS[p] for p in hold_pred],
            "correct": [a == b for a, b in zip(hold_y, hold_pred)],
        }),
    }
    return model, report


# ============================================================================ policy retrieval
class PolicyRetriever:
    """Tiny RAG retriever: TF-IDF over policy articles, boosted by intent mapping."""

    def __init__(self, policies: list[dict]):
        self.policies = policies
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vec.fit_transform([p["title"] + " " + p["text"] for p in policies])

    def search(self, query: str, intent: str | None = None, k: int = 2) -> list[dict]:
        sims = (self.matrix @ self.vec.transform([query]).T).toarray().ravel()
        if intent:
            sims = sims + np.array([0.5 if intent in p["intents"] else 0.0 for p in self.policies])
        top = np.argsort(-sims)[:k]
        best = sims[top[0]] if len(top) else 0
        return [dict(self.policies[i], score=round(float(sims[i]), 3)) for n, i in enumerate(top)
                if sims[i] > 0 and (n == 0 or sims[i] >= 0.5 * best)]


# ============================================================================ analysis
@dataclass
class Analysis:
    message: str
    intent: str
    intent_label: str
    confidence: float
    top_intents: list
    sentiment: str
    sentiment_score: float
    priority: str
    priority_score: int
    priority_reasons: list
    order: dict | None
    customer: dict | None
    product: dict | None
    policies: list
    action: str
    action_detail: str
    escalate: bool
    reply: str = ""
    reply_source: str = "template"
    flags: list = field(default_factory=list)


def sentiment_of(text: str) -> tuple[str, float]:
    t = text.lower()
    score = 0.0
    score -= sum(1 for w in NEG_WORDS if w in t) * 0.6
    score -= sum(1 for w in STRONG_NEG if w in t) * 1.0
    score += sum(1 for w in POS_WORDS if w in t) * 0.5
    score -= min(text.count("!"), 4) * 0.3
    letters = [c for c in text if c.isalpha()]
    if letters and sum(c.isupper() for c in letters) / len(letters) > 0.6 and len(letters) > 15:
        score -= 1.0  # SHOUTING
    if score <= -2:
        return "Angry", score
    if score < -0.4:
        return "Negative", score
    if score > 0.4:
        return "Positive", score
    return "Neutral", score


def _days_since(iso: str) -> int | None:
    if not iso:
        return None
    return (TODAY - date.fromisoformat(iso)).days


def next_best_action(intent: str, order: dict | None, customer: dict | None, product: dict | None):
    """Business rules that turn policy into a concrete recommendation for the agent."""
    gold = bool(customer and customer.get("tier") == "Gold")
    if order is None and intent not in ("product_question", "account_issue"):
        return "Ask for order ID", "No valid order ID found in the message. Ask the customer to share it.", False

    status = order.get("status") if order else None
    if intent == "where_is_my_order":
        if status in ("Processing",):
            return "Share dispatch ETA", f"Order is still processing; promised by {order['promised_delivery']}.", False
        if status in ("Shipped", "Out for Delivery"):
            return "Share tracking link", f"{status} via {order['courier']} (tracking {order['tracking_id']}).", False
        if status == "Delayed":
            return "Apologise + courier escalation", "Order is past its promised date. Raise priority trace.", True
        if status == "Delivered":
            return ("Confirm delivery / raise trace",
                    f"Courier shows delivered on {order['delivered_on']}. "
                    "If customer disputes, open a non-receipt investigation.", False)
        return "Share current status", f"Order status: {status}.", False

    if intent == "delivery_delay":
        late = -_days_since(order["promised_delivery"]) if order["promised_delivery"] else 0
        credit = 250 if gold else 100
        if status == "Delivered":
            return "Clarify status", f"Order was delivered on {order['delivered_on']}. Confirm with customer.", False
        if late <= -3 or status == "Delayed":
            return f"Escalate courier + ₹{credit} credit", "Delay qualifies for compensation under POL-02; "\
                                                           "offer cancellation with full refund as an option.", True
        return "Escalate to courier", "Share revised delivery date after courier escalation.", False

    if intent in ("damaged_item", "wrong_item"):
        since = _days_since(order["delivered_on"]) if order["delivered_on"] else 0
        if intent == "wrong_item":
            return "Reverse pickup + ship correct item", "Free pickup within 48h; priority re-shipment (POL-04).", False
        if since is not None and since <= 7:
            return "Free replacement", f"Reported {since} day(s) after delivery — within 7-day window. "\
                                       "Request 2 photos, then approve replacement.", False
        if product and product.get("warranty_months", 0) > 0:
            return "Route to brand warranty", "Outside 7-day window but within 12-month warranty.", False
        return "Escalate to supervisor", "Outside replacement window; supervisor approval needed.", True

    if intent == "return_request":
        window = int(product["return_window_days"]) if product else 30
        since = _days_since(order["delivered_on"]) if order["delivered_on"] else None
        if status != "Delivered":
            return "Not yet eligible", f"Order is '{status}'. Returns can be raised after delivery.", False
        if window == 0:
            return "Decline politely", "Category is non-returnable once opened (POL-05).", False
        if since is not None and since <= window:
            if product and product["category"] == "Electronics":
                return ("Approve replacement", f"{since} day(s) since delivery, within 7-day window. Electronics: "
                        "replacement only, no change-of-mind returns (POL-05).", False)
            return ("Approve return/exchange", f"{since} day(s) since delivery, within {window}-day window. "
                    "Schedule free pickup.", False)
        return "Outside return window", f"{since} days since delivery exceeds the {window}-day window.", False

    if intent == "refund_status":
        timeline = {"UPI": "1-2", "Wallet": "1-2", "Credit Card": "5-7", "Debit Card": "5-7",
                    "Net Banking": "5-7", "Cash on Delivery": "3-5"}.get(order["payment_method"], "5-7")
        if status in ("Returned", "Cancelled"):
            return "Share refund timeline", f"Refund to {order['payment_method']} takes {timeline} business days "\
                                            f"(₹{order['amount']:,}).", False
        return "Check return status", f"Order status is '{status}' — no refund initiated yet. Guide customer.", False

    if intent == "cancel_order":
        if status == "Processing":
            return "Cancel order", "Not shipped yet — free cancellation, refund per POL-06.", False
        if status in ("Shipped", "Out for Delivery", "Delayed"):
            return "Offer refuse-delivery / return", "Already shipped; cancellation not possible (POL-07).", False
        return "Cannot cancel", f"Order status is '{status}'.", False

    if intent == "payment_issue":
        return "Verify with payment gateway", "Check duplicate debit; refund extra within 48h (POL-08). "\
                                              "Never ask for CVV/OTP.", order is not None and order["amount"] > 5000

    if intent == "product_question":
        return "Answer from product info", "Share specs / warranty from catalogue (POL-09).", False
    if intent == "account_issue":
        return "Guide account recovery", "Follow POL-10 steps; verify identity before changes.", False
    return "Review manually", "", True


def priority_of(sentiment: str, customer, order, text: str, escalate: bool):
    score, reasons = 20, []
    if sentiment == "Angry":
        score += 35; reasons.append("Angry customer")
    elif sentiment == "Negative":
        score += 15; reasons.append("Negative sentiment")
    if customer and customer.get("tier") == "Gold":
        score += 20; reasons.append("Gold member")
    if order and order.get("amount", 0) > 5000:
        score += 15; reasons.append("High-value order (> ₹5,000)")
    if any(w in text.lower() for w in REPEAT_WORDS):
        score += 15; reasons.append("Repeat contact")
    if escalate:
        score += 10; reasons.append("Needs escalation")
    score = min(score, 100)
    label = "P1 - Urgent" if score >= 65 else ("P2 - High" if score >= 40 else "P3 - Normal")
    return label, score, reasons


class SupportPilot:
    def __init__(self, data: dict, model):
        self.d = data
        self.model = model
        self.retriever = PolicyRetriever(data["policies"])
        self.orders = data["orders"].set_index("order_id")
        self.customers = data["customers"].set_index("customer_id")
        self.products = data["products"].set_index("product_id")

    def lookup(self, text: str, customer_id: str | None = None):
        order = customer = product = None
        m = ORDER_RE.search(text)
        if m:
            oid = f"ORD-{m.group(1)}"
            if oid in self.orders.index:
                order = self.orders.loc[oid].to_dict() | {"order_id": oid}
        cid = (order or {}).get("customer_id") or customer_id
        if cid and cid in self.customers.index:
            customer = self.customers.loc[cid].to_dict() | {"customer_id": cid}
        if order and order["product_id"] in self.products.index:
            product = self.products.loc[order["product_id"]].to_dict() | {"product_id": order["product_id"]}
        return order, customer, product

    def analyse(self, text: str, customer_id: str | None = None) -> Analysis:
        probs = self.model.predict_proba([text])[0]
        classes = self.model.classes_
        order_idx = np.argsort(-probs)
        intent = classes[order_idx[0]]
        top = [(INTENT_LABELS[classes[i]], float(probs[i])) for i in order_idx[:3]]

        sentiment, s_score = sentiment_of(text)
        order, customer, product = self.lookup(text, customer_id)
        policies = self.retriever.search(text, intent, k=2)
        action, detail, escalate = next_best_action(intent, order, customer, product)
        if sentiment == "Angry" or any(w in text.lower() for w in ("consumer forum", "legal", "court")):
            escalate = True
        if escalate and all(p["id"] != "POL-11" for p in policies):
            policies.append(next(dict(p, score=0.0) for p in self.d["policies"] if p["id"] == "POL-11"))
        priority, p_score, reasons = priority_of(sentiment, customer, order, text, escalate)

        flags = []
        if probs[order_idx[0]] < 0.5:
            flags.append("Low confidence — agent should verify the intent.")
        if re.search(r"\b(cvv|otp|card number|pin)\b", text.lower()) and intent == "payment_issue":
            flags.append("Security: never request CVV/OTP/PIN from customer.")

        return Analysis(
            message=text, intent=intent, intent_label=INTENT_LABELS[intent], confidence=float(probs[order_idx[0]]),
            top_intents=top, sentiment=sentiment, sentiment_score=s_score, priority=priority,
            priority_score=p_score, priority_reasons=reasons, order=order, customer=customer, product=product,
            policies=policies, action=action, action_detail=detail, escalate=escalate, flags=flags,
        )

    # ------------------------------------------------------------------ reply drafting
    def draft_reply(self, a: Analysis, llm: dict | None = None) -> Analysis:
        if llm and llm.get("api_key"):
            try:
                a.reply = self._llm_reply(a, llm)
                a.reply_source = f"LLM ({llm.get('model')})"
                return a
            except Exception as e:  # never break the demo: fall back to template
                a.flags.append(f"LLM unavailable, used template ({type(e).__name__}).")
        a.reply = self._template_reply(a)
        a.reply_source = "Grounded template"
        return a

    def _llm_reply(self, a: Analysis, llm: dict) -> str:
        from openai import OpenAI  # imported lazily so the app runs without it

        client = OpenAI(api_key=llm["api_key"], base_url=llm.get("base_url") or None)
        context = {
            "customer": {k: a.customer[k] for k in ("name", "tier")} if a.customer else None,
            "order": {k: a.order[k] for k in ("order_id", "product_name", "status", "amount", "payment_method",
                                              "promised_delivery", "delivered_on", "courier", "tracking_id")}
            if a.order else None,
            "policies": [{"title": p["title"], "text": p["text"]} for p in a.policies],
            "recommended_action": f"{a.action}: {a.action_detail}",
            "sentiment": a.sentiment,
        }
        system = (
            "You are a customer-support co-pilot for NovaMart, an Indian e-commerce company. Draft a reply "
            "for the human agent to send. Rules: use ONLY facts from the provided context; never invent "
            "dates, amounts or policies; never ask for card numbers, CVV or OTP; be warm and concise "
            "(80-130 words); apologise sincerely if the customer is upset; end with a clear next step. "
            "Sign off as 'Team NovaMart'."
        )
        resp = client.chat.completions.create(
            model=llm.get("model", "llama-3.3-70b-versatile"),
            temperature=0.3,
            max_tokens=350,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": f"Customer message:\n{a.message}\n\nContext (JSON):\n"
                                                  f"{json.dumps(context, default=str)}"}],
        )
        return resp.choices[0].message.content.strip()

    def _template_reply(self, a: Analysis) -> str:
        name = a.customer["name"].split()[0] if a.customer else "there"
        o = a.order or {}
        prod = o.get("product_name", "your item")
        oid = o.get("order_id", "")
        sorry = ("I'm really sorry for the trouble this has caused, and I completely understand your frustration. "
                 if a.sentiment in ("Angry", "Negative") else "")
        gold = " As a valued Gold member, I've marked your request as priority." if (
            a.customer and a.customer.get("tier") == "Gold") else ""

        if a.action == "Ask for order ID":
            body = ("Could you please share your order ID (it looks like ORD-XXXXX and is in your confirmation "
                    "email)? Once I have it, I'll look into this right away.")
        elif a.intent == "where_is_my_order":
            if o.get("status") in ("Shipped", "Out for Delivery"):
                body = (f"Your {prod} (order {oid}) is currently '{o['status']}' with {o['courier']}. You can track "
                        f"it using tracking ID {o['tracking_id']}. Expected delivery: {o['promised_delivery']}.")
            elif o.get("status") == "Delivered":
                body = (f"Our records show order {oid} ({prod}) was delivered on {o['delivered_on']}. If you haven't "
                        "received it, please let me know and I'll open an investigation with the courier immediately.")
            elif o.get("status") == "Delayed":
                body = (f"Your order {oid} is running behind the promised date. I've escalated it to {o['courier']} on "
                        "priority and will share a revised delivery date within 24 hours.")
            else:
                body = (f"Your order {oid} is being processed and will be dispatched shortly. Expected delivery is "
                        f"by {o.get('promised_delivery', 'the promised date')}.")
        elif a.intent == "delivery_delay":
            credit = "₹250" if a.customer and a.customer.get("tier") == "Gold" else "₹100"
            body = (f"Your order {oid} ({prod}) has missed its promised delivery date of {o.get('promised_delivery')}. "
                    f"I've escalated this with {o.get('courier', 'our courier partner')} on priority"
                    + (f" and added {credit} store credit to your account as an apology" if "credit" in a.action else "")
                    + ". If you'd prefer, I can also cancel the order for a full refund.")
        elif a.intent == "damaged_item":
            if a.action == "Free replacement":
                body = (f"I'm sorry your {prod} arrived damaged. You're eligible for a free replacement or full refund. "
                        "Please reply with 2 photos of the product and packaging, and I'll schedule a pickup and "
                        "dispatch a replacement right away.")
            elif a.action == "Route to brand warranty":
                body = (f"Your {prod} is covered under the 12-month manufacturer warranty. I'll share the nearest "
                        "authorised service centre details and help you register the claim.")
            else:
                body = (f"I've escalated your case for {prod} to my supervisor for a special review and we'll get back "
                        "to you within 24 hours.")
        elif a.intent == "wrong_item":
            body = (f"I'm sorry we sent the wrong item for order {oid}. I've scheduled a free pickup within 48 hours and "
                    f"your {prod} will be shipped on priority at no extra cost.")
        elif a.intent == "return_request":
            if a.action.startswith("Approve"):
                body = (f"Your {prod} (order {oid}) is eligible for {a.action.replace('Approve ', '')}. I've scheduled a "
                        "free pickup from your delivery address. Please keep the item unused with original tags.")
            elif a.action == "Decline politely":
                body = (f"Unfortunately {prod} falls under personal care items, which are non-returnable once opened "
                        "for hygiene reasons. If the product is damaged or defective, I'm happy to help with that.")
            elif a.action == "Not yet eligible":
                body = (f"Your order {oid} hasn't been delivered yet. Once it arrives, you can raise a return from "
                        "the app, or you can refuse the delivery.")
            else:
                body = (f"Your order {oid} is outside the return window for this category. If there's a defect, "
                        "I can help through the warranty route.")
        elif a.intent == "refund_status":
            if a.action == "Share refund timeline":
                days = a.action_detail.split("takes ")[-1].split(" business")[0]
                body = (f"Your refund of ₹{o['amount']:,} for order {oid} has been initiated to your "
                        f"{o['payment_method']}. It usually reflects within {days} business days. If it hasn't "
                        "arrived after that, reply here and I'll trace it with the bank.")
            else:
                body = (f"Order {oid} is currently '{o.get('status')}', so a refund hasn't been initiated yet. Once "
                        "the return/cancellation is completed, the refund is processed within 48 hours.")
        elif a.intent == "cancel_order":
            if a.action == "Cancel order":
                body = (f"I've cancelled order {oid} ({prod}). Your refund of ₹{o['amount']:,} will be processed to "
                        f"your {o['payment_method']} as per our refund timelines.")
            else:
                body = (f"Order {oid} has already been shipped, so it can't be cancelled now. You can simply refuse "
                        "the delivery, or raise a return once it's delivered, and we'll refund you.")
        elif a.intent == "payment_issue":
            body = ("I've raised a verification request with our payment gateway for the extra debit. If a duplicate "
                    "charge is confirmed, it will be refunded within 48 hours. Please note we will never ask for your "
                    "card number, CVV or OTP.")
        elif a.intent == "product_question":
            body = ("All our products are 100% genuine and electronics carry a 12-month manufacturer warranty. "
                    "Detailed specifications are listed on the product page — happy to check any specific detail for you.")
        else:
            body = ("For login issues, please try resending the OTP after 60 seconds or use 'Forgot Password'. Locked "
                    "accounts unlock automatically after 30 minutes. I can also update your details once we verify "
                    "your registered phone number.")

        if "sorry" in body.lower():
            sorry = ""
        return f"Hi {name},\n\n{sorry}{body}{gold}\n\nWarm regards,\nTeam NovaMart"


# ============================================================================ business impact
def impact(tickets_per_day: float, baseline_aht_min: float, aht_reduction: float, agent_monthly_cost: float,
           agent_hours_per_day: float = 8, working_days: int = 26, utilisation: float = 0.75,
           llm_cost_per_ticket: float = 0.40, csat_uplift_pts: float = 0.0) -> dict:
    """
    Cost model:
      agent-hours needed = tickets * AHT / 60 / utilisation
      co-pilot reduces AHT by `aht_reduction`, freeing agent capacity (FTEs)
      savings = FTEs freed * monthly cost per agent - LLM running cost
    """
    monthly_tickets = tickets_per_day * working_days
    hours_before = monthly_tickets * baseline_aht_min / 60 / utilisation
    hours_after = hours_before * (1 - aht_reduction)
    agent_hours_month = agent_hours_per_day * working_days
    fte_before = hours_before / agent_hours_month
    fte_after = hours_after / agent_hours_month
    labour_before = fte_before * agent_monthly_cost
    labour_after = fte_after * agent_monthly_cost
    llm_cost = monthly_tickets * llm_cost_per_ticket
    net_monthly = labour_before - labour_after - llm_cost
    return {
        "monthly_tickets": monthly_tickets,
        "hours_before": hours_before, "hours_after": hours_after, "hours_saved": hours_before - hours_after,
        "fte_before": fte_before, "fte_after": fte_after, "fte_freed": fte_before - fte_after,
        "labour_before": labour_before, "labour_after": labour_after,
        "llm_cost": llm_cost, "net_monthly_saving": net_monthly, "net_annual_saving": net_monthly * 12,
        "roi_multiple": (labour_before - labour_after) / llm_cost if llm_cost else float("inf"),
        "cost_per_ticket_before": labour_before / monthly_tickets,
        "cost_per_ticket_after": (labour_after + llm_cost) / monthly_tickets,
        "aht_after": baseline_aht_min * (1 - aht_reduction),
    }
