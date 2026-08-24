"""Generate the Experiment 001 dataset. Same seed -> byte-identical files."""

import json
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 1234
OUT = Path("data/exp001")

CARRIERS = ["FedEx", "UPS", "DHL", "USPS", "OnTrac"]  # shared across splits on purpose

FIRST = ["Priya", "Marcus", "Yuki", "Ana", "Dilan", "Rosa",
         "Nadia", "Tomas", "Omar", "Elena", "Kwame", "Ingrid"]
LAST = ["Raman", "Webb", "Tanaka", "Ferreira", "Kaya", "Iglesias",
        "Osei", "Vrba", "Haddad", "Novak"]

# Templates are partitioned by split, so a test template cannot leak into train.
TEMPLATES = {
    "train": {
        "T1": "Order #{order_id} for {customer} shipped on {ship_date} via {carrier}.",
        "T2": "{customer} placed order #{order_id}, which was dispatched via {carrier} on {ship_date}.",
        "T3": "On {ship_date}, {carrier} picked up order #{order_id} belonging to {customer}.",
        "T4": "Shipment for order #{order_id} ({customer}) went out {ship_date} with {carrier}.",
        "T5": "{customer}'s order #{order_id} left the warehouse on {ship_date} via {carrier}.",
        "T6": "Customer {customer} - order #{order_id} - shipped {ship_date} by {carrier}.",
    },
    "val": {
        "T7": "{carrier} delivered order #{order_id} to {customer}; ship date was {ship_date}.",
        "T8": "Order #{order_id} was handed to {carrier} on {ship_date} for delivery to {customer}.",
    },
    "test": {
        "T9": "The {carrier} shipment on {ship_date} included order #{order_id} for {customer}.",
        "T10": "Tracking for {customer} shows order #{order_id} shipped {ship_date} through {carrier}.",
    },
}

# n, order-id range, first date, span in days, slice of the shuffled name list
SPLITS = {
    "train": (240, (1000, 5000), date(2024, 1, 1), 365, slice(0, 40)),
    "val": (60, (5000, 7000), date(2025, 1, 1), 90, slice(40, 60)),
    "test": (300, (7000, 10000), date(2025, 4, 1), 275, slice(60, 120)),
}

names = [f"{f} {l}" for f in FIRST for l in LAST]
random.Random(SEED).shuffle(names)

OUT.mkdir(parents=True, exist_ok=True)

for i, (split, (n, id_range, start, span, name_slice)) in enumerate(SPLITS.items()):
    rng = random.Random(SEED + i)  # per-split, so resizing one split doesn't disturb others
    pool = names[name_slice]
    template_ids = list(TEMPLATES[split])
    order_ids = rng.sample(range(*id_range), n)  # unique -> every record is unique

    with open(OUT / f"{split}.jsonl", "w") as f:
        for j, order_id in enumerate(order_ids):
            target = {
                "order_id": str(order_id),
                "customer": rng.choice(pool),
                "ship_date": str(start + timedelta(days=rng.randrange(span))),
                "carrier": rng.choice(CARRIERS),
            }
            tid = template_ids[j % len(template_ids)]  # even coverage per template
            record = {
                "sentence": TEMPLATES[split][tid].format(**target),
                "target": target,
                "template_id": tid,
            }
            f.write(json.dumps(record) + "\n")

    print(f"{split:5s} {n:3d} rows -> {OUT / f'{split}.jsonl'}")
