"""Generate the Experiment 001 dataset. Same seed -> byte-identical files.

Builds 600 synthetic records by filling sentence templates with made-up values.
The point of synthetic data is that ground truth is CONSTRUCTED rather than
labelled, so it is correct by definition - there is no annotation noise to argue
with when a model gets something wrong.

Leakage is prevented structurally rather than checked for afterwards:
  * templates live in three separate dicts, so a test template physically cannot
    appear in train
  * names, order ids and dates come from disjoint pools per split
  * carriers are deliberately SHARED - see the note on CARRIERS below
"""

import json
import random
from datetime import date, timedelta
from pathlib import Path

SEED = 1234
OUT = Path("data/exp001")

# Shared across all splits on purpose. These five are the task's fixed vocabulary,
# like the options in a dropdown - not something the model should generalise past.
# Holding out carriers at test time would quietly change what is being measured.
CARRIERS = ["FedEx", "UPS", "DHL", "USPS", "OnTrac"]

FIRST = ["Priya", "Marcus", "Yuki", "Ana", "Dilan", "Rosa",
         "Nadia", "Tomas", "Omar", "Elena", "Kwame", "Ingrid"]
LAST = ["Raman", "Webb", "Tanaka", "Ferreira", "Kaya", "Iglesias",
        "Osei", "Vrba", "Haddad", "Novak"]

# Templates are partitioned by split, so a test template cannot leak into train.
# Field order varies between templates so a model cannot succeed by position alone
# ("the first number is always the id"). Punctuation oddities - possessives, dashes,
# parentheses - sit in train and val; the two test templates are plainly worded, so a
# drop on test reads as a generalisation failure rather than an unfair difficulty jump.
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

# Per split: row count, order-id range, first date, window length in days, and which
# slice of the shuffled name list to draw from. None of the ranges overlap, so no
# value seen during training can reappear at test time.
# n, order-id range, first date, span in days, slice of the shuffled name list
SPLITS = {
    "train": (240, (1000, 5000), date(2024, 1, 1), 365, slice(0, 40)),
    "val": (60, (5000, 7000), date(2025, 1, 1), 90, slice(40, 60)),
    "test": (300, (7000, 10000), date(2025, 4, 1), 275, slice(60, 120)),
}

# 12 first names x 10 last names = 120 unique full names, shuffled once with a fixed
# seed and then sliced into disjoint pools. Name COMPONENTS recur across splits (train
# may hold "Priya Raman", test "Priya Osei") but full names never do - which suits a
# copy task, since it tests copying novel combinations rather than recalling strings.
names = [f"{f} {l}" for f in FIRST for l in LAST]
random.Random(SEED).shuffle(names)

OUT.mkdir(parents=True, exist_ok=True)

for i, (split, (n, id_range, start, span, name_slice)) in enumerate(SPLITS.items()):
    # Each split gets its own RNG, so changing train's size later cannot shift the
    # contents of the test set - which matters if the data is ever regenerated.
    rng = random.Random(SEED + i)
    pool = names[name_slice]
    template_ids = list(TEMPLATES[split])
    # sample() draws WITHOUT replacement, so order ids are unique within a split,
    # which in turn guarantees every generated record is unique.
    order_ids = rng.sample(range(*id_range), n)

    with open(OUT / f"{split}.jsonl", "w") as f:
        for j, order_id in enumerate(order_ids):
            # Build the answer first, then render the sentence from it. Because the
            # placeholder names match the field names, format(**target) fills the
            # sentence straight from the answer - the two cannot disagree.
            target = {
                "order_id": str(order_id),
                "customer": rng.choice(pool),
                "ship_date": str(start + timedelta(days=rng.randrange(span))),
                "carrier": rng.choice(CARRIERS),
            }
            # Cycle through the split's templates so each is used an equal number of
            # times, rather than letting one phrasing dominate by chance.
            tid = template_ids[j % len(template_ids)]
            record = {
                "sentence": TEMPLATES[split][tid].format(**target),
                "target": target,
                "template_id": tid,
            }
            f.write(json.dumps(record) + "\n")

    print(f"{split:5s} {n:3d} rows -> {OUT / f'{split}.jsonl'}")
