"""Generate the Experiment 001, 002 and 003 datasets. Same seed -> byte-identical files.

Each experiment gets its own directory under data/. exp001 is the frozen baseline;
each later experiment is the previous one plus extra training templates. All of them are
emitted from the same base draw, so every experiment's val/test are byte-identical and
each train.jsonl is the previous one with rows appended - nothing rewritten.

Builds 600 records for exp001, 640 for exp002 and 680 for exp003 by filling sentence
templates with made-up values.
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
OUT_ROOT = Path("data")

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

# One entry per experiment. The value lists ADD-ON rows, appended to a split after its
# base rows are drawn. Each group carries its own seed and its own rng, and its order ids
# come from the ids the base pass did NOT use, so adding or removing a group cannot shift
# a single base row. Every experiment therefore shares one base draw: their val/test files
# are byte-identical, and a later experiment's train.jsonl is an earlier one's with rows
# appended - never rewritten. That is what makes the comparison clean, since any change in
# results is attributable to the added rows and nothing else.
#
# T11 exists to test one thing - a bare sentence-initial carrier, which no exp001 training
# template contains. It is kept structurally apart from val's T7 (single clause, different
# verb, different field order, date cued by "on" and not sentence-final) so a gain on T7
# reads as generalisation rather than memorisation. Note that T11 also breaks the
# ship_date/carrier adjacency that every exp001 template and val's T8 preserve, so position
# and adjacency are varied together, not independently.
#
# T12/T13 target a different gap. Across all 280 exp002 training rows the word "to" is
# followed by the customer 40 times out of 40 and by a carrier never, so "to X" is an
# unambiguous customer-marker - which is exactly what val's T8 violates, and why its
# failures swap the two values. The pair introduces "to" before a CARRIER, in both
# orderings (T12 carrier-first, T13 customer-first) and in equal numbers, so that "first
# to = carrier" is not learnable as a positional shortcut. Passing both requires deciding
# role from what the noun IS - a member of the five-item carrier vocabulary, or an
# open-vocabulary person name - rather than from the preposition.
#
# balanced=True draws carriers from a shuffled fixed schedule (count/5 of each) instead of
# rng.choice, so a per-carrier readout is not confounded by an uneven draw. It is set on
# T12/T13 only; leaving it False everywhere else keeps the earlier rows byte-identical.
# experiment -> split -> [(template id, template, how many, seed, balanced)]
EXPERIMENTS = {
    # Frozen baseline: the six training templates, no add-ons.
    "exp001": {},
    # Training-data diversity: exp001 plus 40 T11 rows, and nothing else.
    "exp002": {
        "train": [
            ("T11", "{carrier} collected order #{order_id} on {ship_date} for {customer}.", 40, 9002, False),
        ],
    },
    # Role disambiguation: exp002 plus 20 T12 + 20 T13 rows, and nothing else.
    "exp003": {
        "train": [
            ("T11", "{carrier} collected order #{order_id} on {ship_date} for {customer}.", 40, 9002, False),
            ("T12", "Order #{order_id} passed to {carrier} and then to {customer} on {ship_date}.", 20, 9003, True),
            ("T13", "On {ship_date}, order #{order_id} was addressed to {customer} and released to {carrier}.", 20, 9004, True),
        ],
    },
}

# 12 first names x 10 last names = 120 unique full names, shuffled once with a fixed
# seed and then sliced into disjoint pools. Name COMPONENTS recur across splits (train
# may hold "Priya Raman", test "Priya Osei") but full names never do - which suits a
# copy task, since it tests copying novel combinations rather than recalling strings.
names = [f"{f} {l}" for f in FIRST for l in LAST]
random.Random(SEED).shuffle(names)


def make_record(rng, tid, template, order_id, pool, start, span, carrier=None):
    """Build one record. The answer is constructed FIRST, then the sentence is rendered
    from it - because the placeholders match the field names, format(**target) fills the
    sentence straight from the answer and the two cannot disagree.

    Passing `carrier` supplies it from a balanced schedule instead of drawing it. That
    also SKIPS the rng.choice(CARRIERS) call, so a balanced group consumes a different
    number of rng values - which is fine, because balanced groups have their own rng."""
    target = {
        "order_id": str(order_id),
        "customer": rng.choice(pool),
        "ship_date": str(start + timedelta(days=rng.randrange(span))),
        "carrier": rng.choice(CARRIERS) if carrier is None else carrier,
    }
    return {"sentence": template.format(**target), "target": target, "template_id": tid}


for exp, extra_groups in EXPERIMENTS.items():
    out = OUT_ROOT / exp
    out.mkdir(parents=True, exist_ok=True)

    for i, (split, (n, id_range, start, span, name_slice)) in enumerate(SPLITS.items()):
        # Each split gets its own RNG, seeded identically for every experiment. That is
        # why the base rows below are shared byte-for-byte across all of them.
        rng = random.Random(SEED + i)
        pool = names[name_slice]
        template_ids = list(TEMPLATES[split])
        # sample() draws WITHOUT replacement, so order ids are unique within a split,
        # which in turn guarantees every generated record is unique.
        order_ids = rng.sample(range(*id_range), n)

        with open(out / f"{split}.jsonl", "w") as f:
            for j, order_id in enumerate(order_ids):
                # Cycle through the split's templates so each is used an equal number of
                # times, rather than letting one phrasing dominate by chance.
                tid = template_ids[j % len(template_ids)]
                record = make_record(rng, tid, TEMPLATES[split][tid], order_id, pool, start, span)
                f.write(json.dumps(record) + "\n")

            # Add-on rows. A fresh rng per group, and ids drawn from the base pass's
            # leftovers, so order ids stay unique within the split and the base rows
            # above are untouched by anything that happens here.
            taken = set(order_ids)
            extra_n = 0
            for tid, template, count, extra_seed, balanced in extra_groups.get(split, []):
                extra_rng = random.Random(extra_seed)
                unused = [oid for oid in range(*id_range) if oid not in taken]
                # An exactly-equal number of each carrier, shuffled so carrier is not
                # correlated with order id. count must divide evenly by len(CARRIERS).
                schedule = None
                if balanced:
                    assert count % len(CARRIERS) == 0, f"{tid}: {count} is not a multiple of {len(CARRIERS)}"
                    schedule = CARRIERS * (count // len(CARRIERS))
                    extra_rng.shuffle(schedule)
                for k, order_id in enumerate(extra_rng.sample(unused, count)):
                    carrier = schedule[k] if schedule else None
                    record = make_record(extra_rng, tid, template, order_id, pool, start, span, carrier)
                    taken.add(order_id)
                    extra_n += 1
                    f.write(json.dumps(record) + "\n")

        suffix = f" (+{extra_n} add-on)" if extra_n else ""
        print(f"{exp} {split:5s} {n + extra_n:3d} rows -> {out / f'{split}.jsonl'}{suffix}")
