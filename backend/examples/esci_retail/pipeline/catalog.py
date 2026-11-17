# SPDX-License-Identifier: Apache-2.0

"""Stage two: the raw ESCI subset to the store's catalog. Standard library only.

ESCI gives a listing's title, description, bullets, brand and color and nothing else, so
the rest is synthesized here: the category from a keyword taxonomy over the title and
bullets; price, rating, review count and stock from a hash of the seed and the product id,
so a listing's figures never depend on which other listings were drawn. The outputs are
``catalog.sqlite`` (one JSON record per product plus two FTS5 indexes: stemmed words for
ranking, trigrams for misspellings), ``queries_gold.json`` (the ESCI labels for the store's
queries), ``orders.json`` (demo order history over in-stock listings) and ``manifest.json``.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import random
import re
import sqlite3
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

from .extract import read_jsonl
from .select import LABELS

GENERAL = "general-merchandise"
CURRENCY = "USD"

# Earlier entries win ties. Each keyword is matched as a whole word or phrase; a phrase
# counts its words, so "coffee maker" outweighs "coffee" and "table" alone.
TAXONOMY: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("pet-supplies", (
        "dog", "dogs", "puppy", "puppies", "cat", "cats", "kitten", "pet", "pets", "leash",
        "litter", "aquarium", "fish tank", "bird cage", "hamster", "kennel", "crate", "dog bed",
        "cat tree", "chew toy", "chew toys", "pet food", "treats",
    )),
    ("baby", (
        "baby", "infant", "newborn", "toddler", "diaper", "diapers", "wipes", "stroller",
        "car seat", "crib", "pacifier", "nursing", "teething", "onesie", "onesies", "bib", "bibs",
        "high chair", "breast pump", "sippy",
    )),
    ("musical-instruments", (
        "guitar", "guitars", "piano", "digital piano", "ukulele", "violin", "drum", "drums",
        "drumsticks", "saxophone", "trumpet", "flute", "amplifier", "amp", "capo", "guitar strings",
        "tuner", "sheet music", "keyboard stand", "midi",
    )),
    ("automotive", (
        "car", "cars", "auto", "automotive", "truck", "trucks", "vehicle", "tire", "tires", "wiper",
        "wipers", "motor oil", "brake", "brakes", "headlight", "headlights", "dash cam", "car charger",
        "car mount", "seat cover", "seat covers", "floor mat", "floor mats", "jump starter", "obd",
        "motorcycle", "rv", "trailer", "hitch", "tonneau", "windshield",
    )),
    ("jewelry-watches", (
        "necklace", "necklaces", "bracelet", "bracelets", "earrings", "earring", "ring", "rings",
        "jewelry", "watch", "watches", "pendant", "charm", "anklet", "cufflinks", "sterling silver",
        "gold plated", "gemstone", "wristwatch",
    )),
    ("bags-luggage", (
        "backpack", "backpacks", "luggage", "suitcase", "suitcases", "duffel", "tote", "handbag",
        "handbags", "purse", "purses", "crossbody", "briefcase", "laptop bag", "fanny pack",
        "travel bag", "carry-on", "carry on", "garment bag", "packing cubes",
    )),
    ("toys-games", (
        "toy", "toys", "lego", "puzzle", "puzzles", "doll", "dolls", "action figure", "action figures",
        "plush", "stuffed animal", "board game", "board games", "card game", "playset", "nerf",
        "rc car", "remote control car", "building blocks", "barbie", "hot wheels", "fidget", "slime",
        "kite", "play kitchen", "dollhouse", "trading cards",
    )),
    ("beauty-personal-care", (
        "shampoo", "conditioner", "hair", "wig", "wigs", "lace front", "makeup", "lipstick", "mascara",
        "eyeshadow", "foundation", "concealer", "nail polish", "nails", "skincare", "moisturizer",
        "serum", "lotion", "sunscreen", "perfume", "cologne", "fragrance", "deodorant", "razor",
        "razors", "shaver", "trimmer", "toothbrush", "toothpaste", "body wash", "soap", "facial",
        "curling iron", "flat iron", "hair dryer", "beard", "eyelash", "eyelashes", "lip gloss",
        "hair extensions", "cosmetic",
    )),
    ("health-household", (
        "vitamin", "vitamins", "supplement", "supplements", "probiotic", "probiotics", "pain relief",
        "first aid", "bandage", "bandages", "thermometer", "blood pressure", "face mask", "face masks",
        "hand sanitizer", "tissues", "paper towels", "toilet paper", "detergent", "cleaner",
        "disinfectant", "allergy", "medicine", "pill organizer", "incontinence", "walker", "cane",
        "bathroom scale", "humidifier", "air purifier", "massager", "heating pad", "compression socks",
        "knee brace", "wheelchair", "glucose", "pregnancy test", "melatonin", "collagen",
        "protein powder",
    )),
    ("grocery", (
        "snack", "snacks", "candy", "candies", "chocolate", "chocolates", "coffee beans",
        "ground coffee", "k-cup", "k-cups", "tea", "tea bags", "cereal", "sauce", "hot sauce", "spice",
        "spices", "seasoning", "flour", "sugar", "olive oil", "honey", "nuts", "almonds", "chips",
        "cookies", "gummies", "gummy", "drink mix", "sparkling water", "juice", "pasta", "rice",
        "beans", "granola", "jerky", "popcorn", "syrup", "gluten free", "keto", "energy drink", "soda",
    )),
    ("sports-outdoors", (
        "fitness", "workout", "yoga", "yoga mat", "dumbbell", "dumbbells", "kettlebell", "weights",
        "weight bench", "resistance band", "resistance bands", "treadmill", "exercise bike", "bike",
        "bicycle", "cycling", "helmet", "camping", "tent", "tents", "sleeping bag", "hiking", "fishing",
        "fishing rod", "golf", "tennis", "basketball", "soccer", "football", "baseball", "softball",
        "skateboard", "scooter", "kayak", "paddle board", "cooler", "flashlight", "headlamp",
        "binoculars", "trampoline", "swim", "swimming", "goggles", "snorkel", "hunting", "archery",
        "boxing", "jump rope", "pickleball", "volleyball", "ski", "snowboard", "water bottle",
        "hydration",
    )),
    ("electronics", (
        "headphones", "headphone", "earbuds", "earbud", "speaker", "speakers", "bluetooth", "wireless",
        "charger", "charging cable", "usb", "usb-c", "hdmi", "laptop", "laptops", "tablet", "ipad",
        "monitor", "monitors", "keyboard", "mouse", "camera", "cameras", "tv", "television", "router",
        "wifi", "phone", "phones", "iphone", "galaxy", "phone case", "screen protector", "smartwatch",
        "smart watch", "drone", "gaming", "console", "controller", "printer", "ink cartridge", "ssd",
        "hard drive", "memory card", "sd card", "projector", "microphone", "webcam", "kindle",
        "e-reader", "power bank", "batteries", "battery", "earphones", "soundbar", "gps", "walkie",
        "radio", "alarm clock", "smart plug", "security camera", "doorbell", "tripod", "flash drive",
        "adapter", "hub", "dock", "docking station", "stylus", "vr headset",
    )),
    ("tools-home-improvement", (
        "drill", "drills", "screwdriver", "screwdrivers", "wrench", "wrenches", "socket set", "hammer",
        "saw", "tool set", "tool kit", "toolbox", "tape measure", "level", "ladder", "nails", "screws",
        "bolts", "plumbing", "faucet", "shower head", "light bulb", "light bulbs", "led bulb",
        "led bulbs", "extension cord", "outlet", "light switch", "thermostat", "door lock", "deadbolt",
        "padlock", "hinge", "hinges", "paint", "caulk", "glue", "epoxy", "sandpaper", "air compressor",
        "generator", "pressure washer", "garage", "work gloves", "safety glasses", "pliers",
        "chainsaw", "welding", "soldering", "multimeter", "ceiling fan", "smoke detector",
        "water filter",
    )),
    ("garden-outdoor", (
        "garden", "gardening", "plant", "plants", "seeds", "planter", "planters", "lawn", "lawn mower",
        "mower", "hose", "garden hose", "sprinkler", "patio", "grill", "grills", "bbq", "barbecue",
        "outdoor furniture", "patio umbrella", "fire pit", "bird feeder", "fence", "mulch",
        "fertilizer", "weed killer", "pest control", "insect", "mosquito", "pool", "snow shovel",
        "leaf blower", "hedge trimmer", "rain barrel", "greenhouse", "trellis", "wind chime",
        "solar lights", "hammock",
    )),
    ("office-school", (
        "office", "desk organizer", "pen", "pens", "pencil", "pencils", "marker", "markers",
        "notebook", "notebooks", "planner", "binder", "binders", "folder", "folders", "stapler",
        "tape dispenser", "labels", "envelopes", "printer paper", "copy paper", "calculator",
        "whiteboard", "sticky notes", "highlighter", "highlighters", "scissors", "clipboard",
        "index cards", "laminator", "shredder", "file cabinet", "desk pad", "mouse pad",
    )),
    ("furniture", (
        "sofa", "couch", "sectional", "chair", "chairs", "armchair", "recliner", "table",
        "dining table", "coffee table", "desk", "standing desk", "bed frame", "mattress", "mattresses",
        "headboard", "dresser", "nightstand", "bookcase", "bookshelf", "cabinet", "ottoman", "bench",
        "stool", "bar stool", "futon", "tv stand", "wardrobe", "shoe rack", "vanity", "loveseat",
        "bunk bed", "office chair", "gaming chair",
    )),
    ("home-kitchen", (
        "kitchen", "cookware", "pan", "pans", "skillet", "pot", "pots", "knife set", "cutting board",
        "blender", "mixer", "stand mixer", "coffee maker", "espresso machine", "kettle", "toaster",
        "air fryer", "microwave", "slow cooker", "pressure cooker", "instant pot", "rice cooker",
        "dishes", "plates", "bowls", "mug", "mugs", "cups", "glasses", "wine glasses", "utensils",
        "spatula", "baking", "bakeware", "storage containers", "food storage", "lunch box", "tumbler",
        "vacuum", "vacuum cleaner", "mop", "broom", "trash can", "towels", "bath towels", "bedding",
        "sheet set", "sheets", "comforter", "duvet", "pillow", "pillows", "blanket", "throw blanket",
        "curtains", "curtain", "rug", "rugs", "lamp", "lamps", "candle", "candles", "decor", "wall art",
        "picture frame", "frames", "shelf", "shelves", "hangers", "organizer", "laundry",
        "laundry basket", "shower curtain", "bath mat", "dish rack", "can opener", "measuring cups",
        "cutlery", "flatware", "placemats", "tablecloth", "ice maker", "water filter pitcher", "fan",
        "space heater", "iron", "ironing board", "sewing machine", "clock", "mirror", "doormat",
        "door mat",
    )),
    ("clothing-shoes", (
        "shirt", "shirts", "t-shirt", "t-shirts", "tshirt", "tee", "tees", "hoodie", "hoodies",
        "sweatshirt", "jacket", "jackets", "coat", "coats", "pants", "jeans", "shorts", "dress",
        "dresses", "skirt", "leggings", "socks", "underwear", "bra", "bras", "boxer briefs", "briefs",
        "sweater", "cardigan", "blouse", "shoes", "sneakers", "boots", "sandals", "slippers", "heels",
        "loafers", "hat", "hats", "cap", "beanie", "gloves", "scarf", "belt", "wallet", "swimsuit",
        "bikini", "pajamas", "pajama", "robe", "uniform", "apparel", "outfit", "outfits", "vest",
        "blazer", "suit", "tie", "sunglasses", "polo", "tank top", "thermal", "jumpsuit", "romper",
        "shapewear", "tights", "lingerie", "flip flops", "cleats", "insoles", "running shoes",
    )),
    ("books-media", (
        "book", "books", "paperback", "hardcover", "novel", "novels", "dvd", "blu-ray", "bluray",
        "vinyl", "vinyl record", "cd", "magazine", "wall calendar", "textbook", "bible", "comic",
        "comics", "manga", "workbook", "coloring book", "cookbook", "journal", "audiobook",
        "boxed set", "video game",
    )),
    ("arts-crafts-party", (
        "craft", "crafts", "yarn", "knitting", "crochet", "sewing", "fabric", "beads", "paint set",
        "acrylic paint", "watercolor", "canvas", "sketchbook", "drawing", "coloring", "stickers",
        "sticker", "party", "balloons", "party supplies", "decorations", "birthday", "wedding",
        "gift wrap", "wrapping paper", "greeting cards", "costume", "costumes", "halloween",
        "christmas", "ornaments", "scrapbook", "glitter", "embroidery", "cricut", "resin", "clay",
        "origami", "pinata", "piñata", "confetti", "banner",
    )),
)  # fmt: skip


def _keyword_pattern(keywords: Iterable[str]) -> re.Pattern[str]:
    ordered = sorted(keywords, key=len, reverse=True)  # longest first, so phrases win
    alternatives = "|".join(re.escape(keyword).replace(r"\ ", r"\s+") for keyword in ordered)
    return re.compile(r"(?<![a-z0-9])(?:" + alternatives + r")(?![a-z0-9])")


_TAXONOMY_PATTERNS = tuple((slug, _keyword_pattern(keywords)) for slug, keywords in TAXONOMY)
_TAG = re.compile(r"<[^>]+>")
_BREAK = re.compile(r"<\s*(?:br|/p|/li|/div|/h\d|/tr)\s*/?>", re.IGNORECASE)
_SPACE = re.compile(r"\s+")
_SENTENCE = re.compile(r"(.+?[.!?])(?:\s|$)")
_SPEC_BULLET = re.compile(r"^([A-Za-z][A-Za-z /&-]{1,28}):\s*(.{2,120})$")

# Price distributions per category: the median in dollars and the log-normal sigma.
PRICE_BANDS: dict[str, tuple[float, float]] = {
    "pet-supplies": (22.0, 0.8),
    "baby": (24.0, 0.9),
    "musical-instruments": (95.0, 1.0),
    "automotive": (32.0, 1.0),
    "jewelry-watches": (36.0, 1.0),
    "bags-luggage": (42.0, 0.8),
    "toys-games": (21.0, 0.8),
    "beauty-personal-care": (15.0, 0.7),
    "health-household": (17.0, 0.8),
    "grocery": (13.0, 0.7),
    "sports-outdoors": (34.0, 1.0),
    "electronics": (42.0, 1.1),
    "tools-home-improvement": (29.0, 1.0),
    "garden-outdoor": (36.0, 1.0),
    "office-school": (13.0, 0.9),
    "furniture": (150.0, 0.8),
    "home-kitchen": (27.0, 0.9),
    "clothing-shoes": (24.0, 0.6),
    "books-media": (15.0, 0.5),
    "arts-crafts-party": (14.0, 0.8),
    GENERAL: (24.0, 1.0),
}


def clean_text(raw: str | None) -> str:
    """HTML to plain text: breaks become spaces, tags go, entities decode, whitespace collapses."""
    if not raw:
        return ""
    text = _TAG.sub(" ", _BREAK.sub(" ", raw))
    return _SPACE.sub(" ", html.unescape(text)).strip()


def bullets_of(raw: str | None, limit: int = 8) -> list[str]:
    """The listing's bullet points (newline-separated in ESCI), cleaned and de-duplicated."""
    seen: set[str] = set()
    bullets: list[str] = []
    for line in (raw or "").split("\n"):
        bullet = clean_text(line)
        if len(bullet) < 3 or bullet.lower() in seen:
            continue
        seen.add(bullet.lower())
        bullets.append(bullet[:300])
        if len(bullets) == limit:
            break
    return bullets


def first_sentence(text: str) -> str:
    match = _SENTENCE.match(text)
    return (match.group(1) if match else text).strip()


def categorize(title: str, bullets: Iterable[str] = ()) -> str:
    """The taxonomy slug whose keywords weigh most in the title (x3) and bullets (x1); a
    phrase counts its words. Ties go to the earlier entry; no hit at all is GENERAL."""
    title_text = title.lower()
    bullet_text = " ".join(bullets).lower()
    best, best_score = GENERAL, 0
    for slug, pattern in _TAXONOMY_PATTERNS:
        score = 3 * sum(len(hit.split()) for hit in pattern.findall(title_text))
        score += sum(len(hit.split()) for hit in pattern.findall(bullet_text))
        if score > best_score:
            best, best_score = slug, score
    return best


def _unit(seed: int, product_id: str, name: str) -> float:
    """A uniform draw in (0, 1) fixed by the seed, the product and the figure's name."""
    digest = hashlib.sha256(f"{seed}|{name}|{product_id}".encode()).digest()
    return (int.from_bytes(digest[:8], "big") + 0.5) / 2**64


def _normal(seed: int, product_id: str, name: str) -> float:
    u1, u2 = _unit(seed, product_id, f"{name}:1"), _unit(seed, product_id, f"{name}:2")
    return math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)


def synthesize(seed: int, product_id: str, category: str) -> dict[str, Any]:
    """Price, rating, review count, stock and labels for one listing, from the seed and id
    alone: the same listing gets the same figures in any build with the same seed."""
    median, sigma = PRICE_BANDS.get(category, PRICE_BANDS[GENERAL])
    price = min(2999.0, max(1.99, median * math.exp(sigma * _normal(seed, product_id, "price"))))
    price = math.floor(price) + 0.99 if price >= 5 else round(price, 2)
    rating = round(min(5.0, max(3.0, 4.35 + 0.35 * _normal(seed, product_id, "rating"))), 1)
    reviews = math.exp(math.log(180) + 1.2 * _normal(seed, product_id, "reviews"))
    review_count = int(min(60_000, max(0, reviews)))
    labels = []
    if _unit(seed, product_id, "new") < 0.06:
        labels.append("new")
        review_count = min(review_count, 40)
    if review_count >= 2500 and rating >= 4.4:
        labels.append("bestseller")
    return {
        "price": price,
        "rating": rating,
        "review_count": review_count,
        "in_stock": _unit(seed, product_id, "stock") < 0.92,
        "labels": labels,
    }


def build_record(raw: Mapping[str, Any], seed: int) -> dict[str, Any]:
    """A ``ProductDetails``-shaped record from one raw ESCI listing."""
    product_id = str(raw["product_id"])
    title = _SPACE.sub(" ", str(raw.get("product_title") or "")).strip()[:200]
    bullets = bullets_of(raw.get("product_bullet_point"))
    description = clean_text(raw.get("product_description"))
    brand = _SPACE.sub(" ", str(raw.get("product_brand") or "")).strip()[:60] or None
    color = _SPACE.sub(" ", str(raw.get("product_color") or "")).strip()[:40] or None
    category = categorize(title, bullets)
    figures = synthesize(seed, product_id, category)
    short = (bullets[0] if bullets else first_sentence(description) or title)[:160]
    long = (description or " ".join(bullets))[:800] or None
    specs: dict[str, str] = {}
    if brand:
        specs["Brand"] = brand
    if color:
        specs["Color"] = color
    for bullet in bullets:
        if (match := _SPEC_BULLET.match(bullet)) and len(specs) < 10:
            specs.setdefault(match.group(1).strip().title(), match.group(2).strip())
    return {
        "product_id": product_id,
        "title": title,
        "brand": brand,
        "price": figures["price"],
        "currency": CURRENCY,
        "rating": figures["rating"],
        "review_count": figures["review_count"],
        "category": category,
        "labels": figures["labels"],
        "attributes": {"color": color.lower()} if color else {},
        "in_stock": figures["in_stock"],
        "short_description": short,
        "long_description": long,
        "specs": specs,
    }


def index_text(raw: Mapping[str, Any], record: Mapping[str, Any]) -> tuple[str, ...]:
    """The row of ``products_fts``: id, title, brand, category words, bullets, description."""
    return (
        record["product_id"],
        record["title"],
        record["brand"] or "",
        record["category"].replace("-", " "),
        " ".join(bullets_of(raw.get("product_bullet_point")))[:1500],
        clean_text(raw.get("product_description"))[:2000],
    )


def gold_entries(judgments: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """One entry per query with the product ids under each ESCI label."""
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in judgments:
        grouped[int(row["query_id"])].append(row)
    entries = []
    for query_id, rows in sorted(grouped.items()):
        entry: dict[str, Any] = {"query_id": query_id, "query": rows[0]["query"]}
        for label in LABELS:
            entry[label] = sorted(r["product_id"] for r in rows if r["esci_label"] == label)
        entries.append(entry)
    return entries


SCHEMA = """
CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    record TEXT NOT NULL,
    category TEXT NOT NULL,
    brand TEXT,
    color TEXT,
    price REAL NOT NULL,
    rating REAL,
    in_stock INTEGER NOT NULL
);
CREATE VIRTUAL TABLE products_fts USING fts5(
    product_id UNINDEXED, title, brand, category, bullets, description,
    tokenize = 'porter unicode61 remove_diacritics 2'
);
CREATE VIRTUAL TABLE products_trigram USING fts5(
    product_id UNINDEXED, title, brand, tokenize = 'trigram'
);
CREATE TABLE queries (query_id INTEGER PRIMARY KEY, query TEXT NOT NULL);
CREATE TABLE judgments (
    query_id INTEGER NOT NULL, product_id TEXT NOT NULL, label TEXT NOT NULL, split TEXT,
    PRIMARY KEY (query_id, product_id)
);
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
"""


def write_sqlite(
    path: Path,
    records: list[dict[str, Any]],
    index_rows: list[tuple[str, ...]],
    judgments: list[dict[str, Any]],
    meta: Mapping[str, Any],
) -> None:
    if path.exists():
        path.unlink()
    db = sqlite3.connect(path)
    try:
        db.executescript(SCHEMA)
        db.executemany(
            "INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [
                (
                    r["product_id"],
                    json.dumps(r, ensure_ascii=False),
                    r["category"],
                    r["brand"],
                    r["attributes"].get("color"),
                    r["price"],
                    r["rating"],
                    int(r["in_stock"]),
                )
                for r in records
            ],
        )
        db.executemany(
            "INSERT INTO products_fts (product_id, title, brand, category, bullets, description) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            index_rows,
        )
        db.executemany(
            "INSERT INTO products_trigram (product_id, title, brand) VALUES (?, ?, ?)",
            [row[:3] for row in index_rows],
        )
        queries = {int(j["query_id"]): j["query"] for j in judgments}
        db.executemany("INSERT INTO queries VALUES (?, ?)", sorted(queries.items()))
        db.executemany(
            "INSERT OR IGNORE INTO judgments VALUES (?, ?, ?, ?)",
            [
                (int(j["query_id"]), j["product_id"], j["esci_label"], j.get("split") or None)
                for j in judgments
            ],
        )
        db.executemany(
            "INSERT INTO meta VALUES (?, ?)", [(k, json.dumps(v)) for k, v in meta.items()]
        )
        db.execute("INSERT INTO products_fts(products_fts) VALUES ('optimize')")
        db.commit()
    finally:
        db.close()


def demo_orders(records: list[dict[str, Any]], seed: int, anchor: date) -> dict[str, Any]:
    """Order history for the two demo profiles over in-stock listings, authored for
    ``anchor`` so the fixture loader's re-dating keeps the estimates current."""
    sellable = sorted((r for r in records if r["in_stock"]), key=lambda r: r["product_id"])
    rng = random.Random(f"orders:{seed}")
    shuffled = rng.sample(sellable, len(sellable))
    picks = (shuffled * 7)[:7] if shuffled else []
    # The delivered order carries one electronics item and one item of another kind, so
    # the two return windows the terms state (15 days opened electronics, 30 days else)
    # both have a line to apply to.
    plainly_electronic = re.compile(
        r"charger|phone|camera|headphone|earbud|speaker|cable|adapter|monitor|keyboard|usb|"
        r"bluetooth|wireless|tablet|laptop|router|printer|drive",
        re.IGNORECASE,
    )
    electronics = [
        r
        for r in shuffled
        if r["category"] == "electronics" and plainly_electronic.search(r["title"])
    ]
    others = [r for r in shuffled if r["category"] != "electronics"]
    if len(picks) == 7 and electronics and others:
        picks[3], picks[4] = electronics[0], others[0]
        used = {picks[3]["product_id"], picks[4]["product_id"]}
        rest = [r for r in shuffled if r["product_id"] not in used]
        for index in (0, 1, 2, 5, 6):
            picks[index] = rest.pop(0) if rest else picks[index]

    def line(record: Mapping[str, Any], quantity: int = 1) -> dict[str, Any]:
        return {
            "product_id": record["product_id"],
            "title": record["title"],
            "quantity": quantity,
            "price": record["price"],
        }

    def placed(days_ago: int, hour: int, minute: int) -> str:
        return f"{(anchor - timedelta(days=days_ago)).isoformat()}T{hour:02d}:{minute:02d}:00Z"

    def on(days: int) -> str:
        return (anchor + timedelta(days=days)).isoformat()

    plan = [
        ("ES-78214", "demo-user", "shipped", placed(8, 14, 32), [(0, 1), (1, 1)], on(1)),
        (
            "ES-77903", "demo-user", "delayed", placed(14, 9, 5), [(2, 1)],
            f"{on(3)} (updated after a carrier delay; the original {on(-3)} estimate was missed)",
        ),
        ("ES-77410", "demo-user", "delivered", placed(23, 19, 47), [(3, 1), (4, 2)], on(-19)),
        ("ES-79236", "demo-user", "shipped", placed(2, 21, 15), [(5, 1)], on(6)),
        ("ES-76988", "demo-user-2", "processing", placed(3, 8, 12), [(6, 2)], on(4)),
    ]  # fmt: skip
    orders = []
    for order_id, user_id, status, placed_at, lines, estimate in plan:
        if not picks:
            break
        items = [line(picks[index], quantity) for index, quantity in lines]
        orders.append(
            {
                "order_id": order_id,
                "user_id": user_id,
                "status": status,
                "placed_at": placed_at,
                "items": items,
                "total": round(sum(item["price"] * item["quantity"] for item in items), 2),
                "estimated_delivery": estimate,
                "tracking_url": f"https://track.example.com/{order_id}",
            }
        )
    return {
        "_comment": (
            "Generated by esci_retail.pipeline from in-stock listings. In-flight orders are "
            "re-dated at load time relative to dates_anchored_to."
        ),
        "dates_anchored_to": anchor.isoformat(),
        "orders": orders,
    }


def build_catalog(
    raw_dir: Path, out_dir: Path, seed: int = 20261106, anchor: date | None = None
) -> dict[str, Any]:
    """``data/raw/`` to the catalog database, the gold file, the order fixture and the
    manifest under ``out_dir``. Returns the manifest."""
    raw_products = sorted(read_jsonl(raw_dir / "products.jsonl"), key=lambda r: r["product_id"])
    judgments = read_jsonl(raw_dir / "examples.jsonl")
    selection_file = raw_dir / "selection.json"
    selection = (
        json.loads(selection_file.read_text(encoding="utf-8")) if selection_file.exists() else {}
    )
    records = [build_record(raw, seed) for raw in raw_products]
    index_rows = [
        index_text(raw, record) for raw, record in zip(raw_products, records, strict=True)
    ]
    anchor = anchor or datetime.now(UTC).date()
    manifest = {
        "built_on": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": seed,
        "orders_anchored_to": anchor.isoformat(),
        "selection": {k: v for k, v in selection.items() if k != "sources"},
        "sources": selection.get("sources", {}),
        "counts": {
            "products": len(records),
            "in_stock": sum(r["in_stock"] for r in records),
            "queries": len({j["query_id"] for j in judgments}),
            "judgments": len(judgments),
            "categories": dict(sorted(Counter(r["category"] for r in records).items())),
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    write_sqlite(out_dir / "catalog.sqlite", records, index_rows, judgments, manifest)
    gold = {"seed": seed, "params": selection.get("params"), "queries": gold_entries(judgments)}
    (out_dir / "queries_gold.json").write_text(
        json.dumps(gold, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (out_dir / "orders.json").write_text(
        json.dumps(demo_orders(records, seed, anchor), indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return manifest
