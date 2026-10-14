"""
Synthetic data generator for SupportPilot.

Creates a realistic (but fully fictional) e-commerce support dataset for
"NovaMart", an Indian online retailer:
  data/products.csv   - catalogue
  data/customers.csv  - customers with loyalty tier
  data/orders.csv     - orders with status, dates, payment info
  data/tickets.csv    - labelled customer-support tickets (intent, baseline handle time)
  data/policies.json  - knowledge-base / policy articles used for grounding replies

Run:  python generate_data.py
"""
import json
import random
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

random.seed(42)
DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)
TODAY = date(2026, 10, 14)

# --------------------------------------------------------------------------- products
CATALOGUE = [
    ("Wireless Earbuds Pro", "Electronics", 2499), ("Smart Fitness Band", "Electronics", 1999),
    ("Bluetooth Speaker Mini", "Electronics", 1499), ("Power Bank 20000mAh", "Electronics", 1299),
    ("USB-C Fast Charger 33W", "Electronics", 899), ("Mechanical Keyboard", "Electronics", 3499),
    ("Noise Cancelling Headphones", "Electronics", 5999), ("Smartwatch Active 2", "Electronics", 4499),
    ("Cotton Kurta Set", "Fashion", 1299), ("Running Shoes Air", "Fashion", 2799),
    ("Denim Jacket", "Fashion", 1899), ("Leather Wallet", "Fashion", 699),
    ("Silk Saree", "Fashion", 3999), ("Sports T-Shirt Pack of 3", "Fashion", 999),
    ("Non-stick Cookware Set", "Home & Kitchen", 2299), ("Air Fryer 4L", "Home & Kitchen", 4999),
    ("Steel Water Bottle 1L", "Home & Kitchen", 449), ("Ceramic Dinner Set", "Home & Kitchen", 2599),
    ("Mixer Grinder 750W", "Home & Kitchen", 3299), ("LED Desk Lamp", "Home & Kitchen", 1099),
    ("Bedsheet Double King", "Home & Kitchen", 1199), ("Face Serum Vitamin C", "Beauty", 599),
    ("Hair Dryer Compact", "Beauty", 1399), ("Perfume Oud 100ml", "Beauty", 1799),
    ("Yoga Mat 6mm", "Sports", 799), ("Dumbbell Set 10kg", "Sports", 1999),
    ("Cricket Bat Kashmir Willow", "Sports", 1599), ("Kids Story Book Set", "Books & Toys", 899),
    ("Building Blocks 500pcs", "Books & Toys", 1499), ("Office Backpack 30L", "Fashion", 1699),
]
products = pd.DataFrame(
    [{"product_id": f"P{100 + i}", "product_name": n, "category": c, "price": p,
      "return_window_days": 7 if c == "Electronics" else (0 if c == "Beauty" else 30),
      "warranty_months": 12 if c == "Electronics" else 0}
     for i, (n, c, p) in enumerate(CATALOGUE)]
)

# --------------------------------------------------------------------------- customers
FIRST = ["Aarav", "Vivaan", "Aditya", "Ishaan", "Rohan", "Karan", "Arjun", "Rahul", "Priya", "Ananya",
         "Diya", "Sneha", "Pooja", "Neha", "Kavya", "Meera", "Riya", "Sanjay", "Vikram", "Amit",
         "Nikhil", "Sakshi", "Tanvi", "Harsh", "Manish", "Divya", "Simran", "Gaurav", "Aisha", "Farhan"]
LAST = ["Sharma", "Verma", "Gupta", "Singh", "Kumar", "Patel", "Reddy", "Iyer", "Nair", "Mehta",
        "Joshi", "Chopra", "Malhotra", "Bose", "Das", "Khan", "Agarwal", "Kapoor", "Rao", "Yadav"]
CITIES = ["Gurugram", "Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Pune", "Chennai", "Kolkata",
          "Jaipur", "Lucknow", "Ahmedabad", "Chandigarh", "Noida", "Indore", "Kochi"]

customers = []
for i in range(250):
    fn, ln = random.choice(FIRST), random.choice(LAST)
    tier = random.choices(["Gold", "Silver", "Regular"], weights=[15, 30, 55])[0]
    customers.append({
        "customer_id": f"C{1000 + i}",
        "name": f"{fn} {ln}",
        "email": f"{fn.lower()}.{ln.lower()}{random.randint(1, 99)}@example.com",
        "city": random.choice(CITIES),
        "tier": tier,
        "lifetime_orders": random.randint(1, 60) + (25 if tier == "Gold" else 0),
        "customer_since": (TODAY - timedelta(days=random.randint(30, 1800))).isoformat(),
    })
customers = pd.DataFrame(customers)

# --------------------------------------------------------------------------- orders
STATUSES = ["Processing", "Shipped", "Out for Delivery", "Delivered", "Delivered", "Delivered",
            "Delayed", "Cancelled", "Returned"]
PAYMENTS = ["UPI", "Credit Card", "Debit Card", "Cash on Delivery", "Net Banking", "Wallet"]
COURIERS = ["BlueDart", "Delhivery", "Ekart", "XpressBees", "Shadowfax"]

orders = []
for i in range(600):
    prod = products.sample(1, random_state=random.randint(0, 10**6)).iloc[0]
    cust = customers.sample(1, random_state=random.randint(0, 10**6)).iloc[0]
    status = random.choice(STATUSES)
    if status in ("Processing", "Shipped", "Out for Delivery"):
        order_date = TODAY - timedelta(days=random.randint(0, 3))
    else:
        order_date = TODAY - timedelta(days=random.randint(4, 40))
    promised = order_date + timedelta(days=random.randint(4, 7))
    delivered = None
    if status in ("Delivered", "Returned"):
        delivered = min(TODAY, order_date + timedelta(days=random.randint(2, 6)))
    elif status == "Delayed":
        promised = TODAY - timedelta(days=random.randint(1, 4))
    qty = random.choices([1, 1, 1, 2, 3], k=1)[0]
    orders.append({
        "order_id": f"ORD-{50000 + i}",
        "customer_id": cust.customer_id,
        "product_id": prod.product_id,
        "product_name": prod.product_name,
        "category": prod.category,
        "quantity": qty,
        "amount": int(prod.price * qty),
        "payment_method": random.choice(PAYMENTS),
        "order_date": order_date.isoformat(),
        "promised_delivery": promised.isoformat(),
        "delivered_on": delivered.isoformat() if delivered else "",
        "status": status,
        "courier": random.choice(COURIERS),
        "tracking_id": f"TRK{random.randint(10**8, 10**9 - 1)}",
    })
orders = pd.DataFrame(orders)

# --------------------------------------------------------------------------- tickets
# intent -> (baseline handle time in minutes without co-pilot, list of templates)
INTENTS = {
    "where_is_my_order": (6, [
        'Where is my parcel? Order {oid}', 'Any update on my shipment {oid}?', 'When will my package be delivered, order {oid}', 'What is the expected delivery date for {oid}?', 'Is my {prod} out for delivery yet? {oid}', 'Order {oid} - shipping status?', 'Need the delivery ETA for my order {oid}', 'Mera order {oid} kahan hai?',
        "Where is my order {oid}? It still hasn't arrived.",
        "Hi, can you tell me the status of order {oid}?",
        "I ordered {prod} last week, order {oid}, when will it be delivered?",
        "Tracking for {oid} is not updating. Where is my package?",
        "please share delivery update for {oid}",
        "My {prod} has not reached yet, order id {oid}. Kab tak aayega?",
        "Can I get the tracking details for my {prod}? Order {oid}",
        "Has my order {oid} been shipped yet?",
    ]),
    "delivery_delay": (9, [
        'Delivery date missed for {oid}, still waiting', 'Late delivery again for {oid}! It should have come on {date}', 'Why is it taking so long? {oid} was due {n} days ago', 'Your courier is too slow, {oid} is overdue', 'Order {oid} is very late, no delivery yet',
        "Order {oid} was supposed to arrive on {date} and it's still not here. This is ridiculous!",
        "My delivery is late by {n} days. Order {oid}. Very disappointed.",
        "Why is my {prod} delayed?? It was promised by {date}. Order {oid}",
        "The courier keeps postponing delivery of {oid}. I needed it urgently for a gift.",
        "Expected delivery date has passed for {oid}. No update from your side at all.",
        "It's been {n} days past the promised date for {oid}. Fix this now.",
    ]),
    "damaged_item": (12, [
        '{prod} not working out of the box. Order {oid}', 'The {prod} is faulty, it does not switch on. {oid}', 'Received {prod} with missing parts and dents, order {oid}', 'My {prod} is dead on arrival. {oid}', 'Product quality defect - {prod} leaking/broken. {oid}',
        "I received my {prod} today but it is damaged. Order {oid}.",
        "The {prod} arrived broken, box was crushed. Order {oid}. I want a replacement.",
        "Product in order {oid} is defective, {prod} is not turning on.",
        "My {prod} came with scratches and a cracked part. Order {oid}",
        "Received a damaged {prod}, really upset. Please help, order {oid}",
        "The {prod} stopped working after 2 days. Order {oid}. Need replacement asap.",
    ]),
    "wrong_item": (11, [
        'Different product shipped instead of {prod}. {oid}', 'Received wrong colour of {prod}, I ordered another one. {oid}', 'This is not my order, item mismatch for {oid}', 'You delivered another model, not {prod}. Order {oid}',
        "I ordered {prod} but received something completely different. Order {oid}",
        "Wrong product delivered for order {oid}. This is not what I ordered.",
        "You sent me the wrong size/colour for {prod}, order {oid}.",
        "Got someone else's item in my package for {oid}. Where is my {prod}?",
        "Order {oid}: item received does not match the listing of {prod}.",
    ]),
    "return_request": (8, [
        '{prod} is too small, I want to send it back. Order {oid}', 'Want to return {prod}, not happy with it. {oid}', 'Please take back the {prod}, size does not fit. {oid}', 'Initiate return pickup for {oid}', 'Exchange {prod} for bigger size, order {oid}',
        "I want to return the {prod} from order {oid}. It doesn't fit well.",
        "How do I return order {oid}? I don't like the {prod}.",
        "Please arrange a pickup for return of {prod}, order {oid}.",
        "Can I return {prod}? I changed my mind. Order {oid}",
        "I'd like to exchange the {prod} in order {oid} for a different size.",
        "Return request for {oid}, quality of {prod} is not as expected.",
    ]),
    "refund_status": (8, [
        'Money not received after return of {oid}', 'Refund still not credited to my bank for {oid}', 'When will refund for {oid} reflect in my account?', 'Pickup done {n} days ago but no money back for {oid}', 'Where is my money for the returned {prod}? {oid}',
        "I returned order {oid} but haven't got my refund yet.",
        "When will I get my refund for {oid}? It's been {n} days.",
        "Refund for cancelled order {oid} not credited to my account.",
        "My money for {oid} has not come back. Where is my refund??",
        "Refund status please for order {oid}. Paid via {pay}.",
        "Still waiting for refund of Rs {amt} for {oid}. Please check.",
    ]),
    "cancel_order": (5, [
        'Stop my order {oid}, I do not need it anymore', "Don't ship order {oid}, please cancel", 'I placed {oid} by mistake, cancel it', 'Cancel the {prod} I ordered today, {oid}',
        "Please cancel my order {oid}.",
        "I want to cancel {oid}, ordered {prod} by mistake.",
        "Cancel order {oid} immediately, found it cheaper elsewhere.",
        "Can I still cancel order {oid}? I no longer need the {prod}.",
        "Kindly cancel {oid} and refund the amount.",
    ]),
    "payment_issue": (10, [
        'I was billed twice for {oid}', 'Charged two times on my {pay} for {oid}', 'Amount deducted but order not placed, {oid}', 'Extra money debited from account for {oid}', 'Payment debited but showing failed in app. {oid}',
        "Money got deducted twice for order {oid}. Please refund the extra amount.",
        "Payment failed but amount was debited via {pay}. Order {oid}",
        "I was charged Rs {amt} but order {oid} shows payment pending.",
        "My {pay} payment for {oid} went through but no order confirmation.",
        "Double charge on my card for {oid}! Need this fixed today.",
    ]),
    "product_question": (5, [
        'What are the specifications of {prod}?', 'Is the {prod} waterproof?', 'How long is the warranty on {prod}?', 'Does {prod} have bluetooth?', 'What material is {prod} made of?', 'Will {prod} fit in a small kitchen? Dimensions?',
        "Does the {prod} come with a warranty?",
        "Is the {prod} compatible with iPhone?",
        "What is the size/material of the {prod}?",
        "Is {prod} available in other colours?",
        "Can you tell me if {prod} supports fast charging?",
        "Is the {prod} original? Want to know before buying.",
    ]),
    "account_issue": (7, [
        'OTP not received while signing in', 'Unable to login to my NovaMart account', 'How to change email id in my profile?', 'Forgot password, reset not working', 'Need to update my mobile number in profile', 'My account got blocked',
        "I can't log in to my account, OTP is not coming.",
        "Please update my phone number on my account.",
        "My account is locked after too many attempts.",
        "How do I change my delivery address saved in the account?",
        "Unable to reset password, link expired.",
        "I want to delete my account and data.",
    ]),
}

OPENERS = ["", "", "", "Hi, ", "Hello team, ", "Dear NovaMart, ", "Hey, ", "Namaste, "]
CLOSERS = ["", "", "", " Thanks.", " Please help.", " Regards.", " Waiting for your reply.",
           " This is the second time I'm writing!", " Pls revert asap."]


def noisy(text: str) -> str:
    """Add light real-world noise: casing, missing punctuation, typos."""
    r = random.random()
    if r < 0.12:
        text = text.lower()
    elif r < 0.17:
        text = text.upper()
    if random.random() < 0.15:
        text = text.replace(".", "").replace("?", "")
    if random.random() < 0.10:
        text = text.replace("order", "ordr").replace("please", "pls")
    return text


INTENT_STATUS = {
    "where_is_my_order": ["Processing", "Shipped", "Out for Delivery", "Delayed", "Delivered"],
    "delivery_delay": ["Delayed"],
    "damaged_item": ["Delivered"], "wrong_item": ["Delivered"], "return_request": ["Delivered"],
    "refund_status": ["Returned", "Cancelled"],
    "cancel_order": ["Processing", "Processing", "Shipped"],
}
tickets = []
channels = ["Email", "Chat", "Chat", "WhatsApp", "App"]
for i in range(1200):
    intent = random.choices(list(INTENTS), weights=[22, 10, 10, 6, 12, 12, 8, 7, 8, 5])[0]
    base_aht, templates = INTENTS[intent]
    pool = orders[orders.status.isin(INTENT_STATUS[intent])] if intent in INTENT_STATUS else orders
    order = pool.sample(1, random_state=random.randint(0, 10**6)).iloc[0]
    body = random.choice(templates).format(
        oid=order.order_id, prod=order.product_name, date=order.promised_delivery,
        n=random.randint(2, 9), pay=order.payment_method, amt=order.amount)
    body = noisy(random.choice(OPENERS) + body + random.choice(CLOSERS))
    has_order = "{oid}" in " ".join(templates) and order.order_id in body
    created = TODAY - timedelta(days=random.randint(0, 29), hours=random.randint(0, 23))
    tickets.append({
        "ticket_id": f"TCK-{80000 + i}",
        "created_at": created.isoformat(),
        "channel": random.choice(channels),
        "customer_id": order.customer_id if has_order else customers.sample(
            1, random_state=random.randint(0, 10**6)).iloc[0].customer_id,
        "order_id": order.order_id if has_order else "",
        "message": body,
        "intent": intent,
        # baseline handle time (minutes) for a human agent WITHOUT co-pilot, with variation
        "baseline_handle_min": round(max(2, random.gauss(base_aht, base_aht * 0.25)), 1),
    })
tickets = pd.DataFrame(tickets)

# --------------------------------------------------------------------------- policies (knowledge base)
policies = [
    {"id": "POL-01", "title": "Order Tracking & Delivery Timelines", "intents": ["where_is_my_order"],
     "text": "Orders are dispatched within 24-48 hours. Metro cities: 2-4 days, other cities: 4-7 days. "
             "Customers can track orders with the tracking ID on the courier partner's website. "
             "If tracking has not updated for 48 hours, agent should raise a courier trace request."},
    {"id": "POL-02", "title": "Late Delivery Compensation", "intents": ["delivery_delay"],
     "text": "If an order is delayed beyond the promised delivery date, apologise, escalate to the courier "
             "with priority, and share a revised date. For delays of 3+ days, issue a Rs 100 store credit "
             "(Rs 250 for Gold members). Customer may cancel a delayed undelivered order for a full refund."},
    {"id": "POL-03", "title": "Damaged or Defective Items", "intents": ["damaged_item"],
     "text": "Damaged/defective items reported within 7 days of delivery qualify for a free replacement or "
             "full refund. Ask the customer for 2 photos of the product and packaging. Electronics within "
             "warranty (12 months) after 7 days are handled via brand service centre."},
    {"id": "POL-04", "title": "Wrong Item Delivered", "intents": ["wrong_item"],
     "text": "If a wrong item is delivered, schedule a free reverse pickup within 48 hours and ship the "
             "correct item on priority at no extra cost. Refund is offered if the correct item is out of stock."},
    {"id": "POL-05", "title": "Return & Exchange Policy", "intents": ["return_request"],
     "text": "Fashion, Home and Sports items: 30-day return/exchange. Electronics: 7-day replacement only "
             "(no change-of-mind returns). Beauty and personal care items are non-returnable once opened. "
             "Items must be unused with original tags. Free pickup is arranged from the delivery address."},
    {"id": "POL-06", "title": "Refund Timelines", "intents": ["refund_status", "cancel_order"],
     "text": "Refunds are initiated once the returned item passes quality check (within 48 hours of pickup). "
             "UPI/Wallet: 1-2 business days. Cards/Net Banking: 5-7 business days. Cash on Delivery orders "
             "are refunded to NovaMart Wallet instantly or to bank account in 3-5 business days."},
    {"id": "POL-07", "title": "Order Cancellation", "intents": ["cancel_order"],
     "text": "Orders can be cancelled free of charge before they are shipped. Once shipped, the customer can "
             "refuse delivery or raise a return after delivery as per return policy."},
    {"id": "POL-08", "title": "Payment Failures & Double Charges", "intents": ["payment_issue"],
     "text": "If money is debited but the order is not confirmed, the amount auto-reverses within 5-7 "
             "business days. Duplicate charges are verified with the payment gateway and refunded within "
             "48 hours. Agents must never ask for full card numbers, CVV or OTP."},
    {"id": "POL-09", "title": "Product Information & Warranty", "intents": ["product_question"],
     "text": "All products are 100% genuine and sourced from brands or authorised sellers. Electronics carry "
             "a 12-month manufacturer warranty. Detailed specifications are on the product page."},
    {"id": "POL-10", "title": "Account & Login Help", "intents": ["account_issue"],
     "text": "For OTP issues, verify the registered phone number and resend OTP after 60 seconds. Locked "
             "accounts unlock automatically after 30 minutes or via the 'Forgot Password' link. Account "
             "deletion requests are processed within 7 days under the privacy policy."},
    {"id": "POL-11", "title": "Escalation & Priority Handling", "intents": [],
     "text": "Escalate to a Tier-2 supervisor when: the customer is very angry, mentions legal/consumer forum, "
             "has contacted 2+ times, or the order value is above Rs 5,000. Gold members get priority queue."},
]

products.to_csv(DATA / "products.csv", index=False)
customers.to_csv(DATA / "customers.csv", index=False)
orders.to_csv(DATA / "orders.csv", index=False)
tickets.to_csv(DATA / "tickets.csv", index=False)
(DATA / "policies.json").write_text(json.dumps(policies, indent=2))

print(f"products={len(products)} customers={len(customers)} orders={len(orders)} tickets={len(tickets)}")
print(tickets.intent.value_counts())
