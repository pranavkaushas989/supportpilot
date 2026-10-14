# SupportPilot — Presentation & Viva Notes

## The one-line pitch
"SupportPilot is an AI co-pilot that cuts customer-support handle time by ~25%, freeing about 11 agents and
saving roughly ₹43 lakh a year for a company handling 2,000 tickets a day, while keeping a human in control."

## 5-minute demo script

1. **Problem (Overview page, 45 s)** — Agents spend ~8 minutes per ticket, mostly searching the order system and
   policy docs. Replies are inconsistent, and angry high-value customers wait in the same queue as simple questions.
2. **Reference case (15 s)** — Wayfair built a similar co-pilot (ZenML LLMOps Database) and saw ~10% lower handle time.
3. **Live demo (Agent Workspace, 90 s)**
   - Choose *Write my own* → the headphones message (ORD-50358) → **Analyse**.
   - Point out: intent detected, negative sentiment, High priority because the order is above ₹5,000, the order
     record pulled automatically, policy POL-03 retrieved, recommended action "Free replacement", draft reply ready.
   - Edit one line, click **Approve & send** → session minutes-saved counter goes up.
   - Bonus: type an angry message mentioning "consumer forum" → it becomes P1 and is flagged for escalation.
4. **Smart queue (30 s)** — 150 tickets triaged in under a second vs ~150 minutes manually; urgent ones on top.
5. **Business impact (90 s)** — walk the cost model, change tickets/day live, show the 3 scenarios and success criteria.
6. **Trust (30 s, Model Evaluation)** — ~97% accuracy on messages written in new words, human-in-the-loop, fallback.

## The impact numbers (expected scenario)

| Assumption | Value |
|---|---|
| Tickets per day | 2,000 (26 working days → 52,000 / month) |
| Manual handle time | ~7.8 min (average of our labelled data) |
| Agent cost | ₹35,000 / month fully loaded |
| Agent utilisation | 75% |
| AI cost | ₹0.40 per ticket |
| Handle-time reduction | 25% (conservative 10% = Wayfair's reported figure, optimistic 40%) |

Result: ~2,250 agent-hours saved per month ≈ 10.8 agents of capacity ≈ **₹3.6 lakh/month, ₹43 lakh/year net**,
AI costs only ~₹21,000/month (≈18× return on AI spend). Cost per ticket falls from ~₹29 to ~₹22.

## Success criteria
- AHT down ≥ 25% (A/B test: co-pilot agents vs control group for 4 weeks)
- ≥ 60% of drafts sent with no or minor edits
- Intent accuracy ≥ 90% on real tickets
- P1 first response < 15 minutes
- Repeat contacts down 15%, CSAT up 5 points
- 100% of replies consistent with the retrieved policy

## Likely questions and answers

**Why not let the AI reply directly to customers?**
Refunds and compensation have money and legal risk. A co-pilot keeps a human accountable, which makes adoption
easier and avoids wrong promises. Fully automating simple "where is my order" tickets is the next phase.

**How do you know the 25% is realistic?**
Wayfair reported ~10% from reply suggestions alone. We also automate order lookup, policy search and triage,
so 25% is our expected case, but we show 10% as the conservative case and would validate with an A/B pilot.

**Isn't 100% test accuracy suspicious?**
Yes — the test split comes from the same synthetic generator. That's why we also report ~97% on 31 hand-written
messages in completely new wording. With real data we would expect lower accuracy initially and retrain weekly.

**What if the LLM makes something up (hallucination)?**
The LLM only receives the order record and retrieved policy, is told to use only those facts, the agent reviews
every draft, and the app falls back to grounded templates if the API fails.

**Will people lose jobs?**
Freed capacity is usually used to absorb festive-season peaks and growth without new hiring, and to move agents to
higher-value work like retention and sales.

**Why TF-IDF + Logistic Regression, not an LLM, for intent?**
It's fast (~3 ms per ticket), free to run, explainable and accurate enough for 10 well-defined intents. The LLM is
used only where language generation adds value — the reply.

**What would you change for production?**
Connect to Zendesk/Freshdesk and the order system via API, embeddings + vector DB for policies, feedback loop from
agent edits, monitoring of acceptance rate, cost and latency, and data-privacy controls (mask PII before the LLM).
