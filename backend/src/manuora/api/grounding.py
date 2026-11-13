"""Optional answer-grounding checks run after a draft passes the citation rules.

Both are off by default (config/assistant.yaml `grounding`); experiment O measures them.
- term check: every name the question uses (a product, service, file type, menu, identifier)
  must occur in the evidence; a manual that never says "Google Drive" cannot support an answer about it.
- support check: one short call to the same chat model asks whether the evidence states the answer.
"""

from __future__ import annotations

import re

# Capitalised words that start or glue questions and are not names.
COMMON = {
    "a", "an", "and", "are", "can", "could", "do", "does", "for", "from", "how", "i", "if", "in", "is", "it",
    "my", "on", "or", "our", "please", "should", "the", "there", "this", "to", "we", "what", "when", "where",
    "which", "who", "why", "will", "with", "would", "after", "before", "also", "any", "all", "is", "me",
    "yes", "no", "hi", "hello", "thanks", "ok", "okay",
}
# Product ids and display names. Defaults cover the public manuals; config `products:` adds more
# (register_products), so local-only manuals never need to be named in code.
PRODUCT_NAMES = {"nextcloud": "Nextcloud", "qgis": "QGIS", "kicad": "KiCad", "freecad": "FreeCAD"}
PRODUCTS = set(PRODUCT_NAMES) | {"pyqgis"}


def register_products(names: dict[str, str]) -> None:
    """Add {product_id: display name}; product names are never treated as terms the evidence must contain."""
    for product_id, display in (names or {}).items():
        PRODUCT_NAMES[str(product_id)] = str(display)
        PRODUCTS.add(str(product_id).casefold())
        PRODUCTS.add(str(display).casefold())

_QUOTED = re.compile(r"[\"“'‘]([^\"”'’]{2,60})[\"”'’]")
# identifiers: occ files:scan, $area, selectedFeatures(), .rvt, config.php, snake_case
_IDENT = re.compile(r"(?<!\w)(?:\$\w+|\.\w{2,5}\b|\w+(?:[:_.]\w+)+|\w+\(\))")
_WORD = re.compile(r"(?<![\w$.])\w[\w\-+]*")  # whole tokens, so "3D" stays "3D"


def _is_name(word: str, sentence_start: bool) -> bool:
    if word.casefold() in COMMON or word.casefold() in PRODUCTS or not any(c.isalpha() for c in word):
        return False
    inner_caps = any(c.isupper() for c in word[1:])  # LDAP, DVH, PyQGIS, CamelCase
    has_digit = any(c.isdigit() for c in word)  # S3, EQD2
    if inner_caps or (has_digit and word[0].isalpha() and any(c.isupper() for c in word)):
        return True
    return word[0].isupper() and not sentence_start


_HOST_END = {"com", "org", "net", "io", "de", "uk", "fr", "eu", "local", "lan", "internal", "example", "test", "home"}


def _identifier(token: str) -> str | None:
    """Keep manual identifiers; drop the user's own hosts and addresses; reduce file names to their extension."""
    if re.fullmatch(r"\d+(\.\d+)*", token) or "@" in token:
        return None  # version numbers, IP addresses
    if "." in token and not token.startswith(".") and ":" not in token and "_" not in token and "(" not in token:
        parts = token.split(".")
        if len(parts) > 2 or parts[-1].casefold() in _HOST_END:
            return None  # cloud.example.com, alpha.example
        return "." + parts[-1]  # ports.shp -> .shp (the user's file name is not a manual term)
    return token


def named_terms(question: str) -> list[str]:
    """Names the question relies on: quoted text, identifiers, and runs of capitalised words."""
    terms = [m.group(1).strip() for m in _QUOTED.finditer(question)]
    terms += [t for t in (_identifier(m.group(0)) for m in _IDENT.finditer(question)) if t]
    for sentence in re.split(r"(?<=[.?!])\s+|\n+", question):
        run, start = [], True
        for m in _WORD.finditer(sentence):
            word = m.group(0)
            if _is_name(word, start):
                run.append(word)
            elif run:
                terms.append(" ".join(run)); run = []
            start = False
        if run:
            terms.append(" ".join(run))
    seen, out = set(), []
    for term in terms:
        key = term.casefold()
        if key not in seen and key not in PRODUCTS:
            seen.add(key); out.append(term)
    return out


def _norm(text: str) -> str:
    return " " + re.sub(r"[^\w$.:()]+", " ", text.casefold()) + " "


def _stem(word: str) -> str:
    return word[:-1] if len(word) > 4 and word.endswith("s") else word


def compact(text: str) -> str:
    """Spelling-insensitive form: lowercase letters and digits only, so PartDesign = Part Design = part-design."""
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


def term_present(term: str, texts: list[str], spelling: bool = False) -> bool:
    """The whole term, or every word of it (plural-insensitive) inside one passage.

    spelling=True adds one more way to match, never fewer: the compact form (case, spaces, hyphens and
    apostrophes ignored, plural tolerated) for terms of 5+ characters, so "PartDesign" finds "Part Design".
    """
    words = [_stem(w) for w in _norm(term).split()]
    key = compact(term)
    key = key[:-1] if len(key) > 5 and key.endswith("s") else key
    for text in texts:
        norm = _norm(text)
        if _norm(term).strip() in norm or all(w in norm for w in words):
            return True
        if spelling and len(key) >= 5 and key in compact(text):
            return True
    return False


def passage_text(hit: dict) -> str:
    """What the model sees of a passage: its heading path and its text."""
    return " / ".join(str(h) for h in (hit.get("headings") or [])) + "\n" + str(hit.get("text") or "")


def missing_terms(question: str, texts: list[str], spelling: bool = False) -> list[str]:
    return [t for t in named_terms(question) if not term_present(t, texts, spelling)]


# --------------------------------------------------------------------------- version check
_FEATURE_STOP = {
    "a", "an", "and", "the", "of", "to", "in", "on", "for", "with", "by", "from", "your", "my", "how", "do",
    "is", "are", "what", "can", "use", "using", "new", "create", "add", "open", "set", "up", "into", "via",
    "at", "as", "or", "this", "that", "it", "be", "i", "we", "our", "you", "not", "no", "all", "more", "other",
}
def _words(text: str) -> list[str]:
    return [_stem(w) for w in re.sub(r"[^a-z0-9]+", " ", text.casefold()).split()]


class VersionIndex:
    """Which manual versions mention a product feature (config `grounding.version_check`; experiment P).

    Features are 2-3 word phrases taken from section headings of every manual of a product, so they are
    the manuals' own names ("mesh layer", "data source manager"). A question that uses such a phrase,
    for a version whose manual never mentions it while another version's manual does, asks about a
    feature from another version. Presence is a phrase query on the full-text index (chunk_fts).
    """

    def __init__(self, sqlite_path, source: str = "headings", min_elsewhere: int = 3):
        """source: "headings" (feature phrases from section headings) or "text" (any question phrase that
        at least `min_elsewhere` passages of another version mention)."""
        import json
        import sqlite3

        if source not in ("headings", "text"):
            raise ValueError("version_check source must be 'headings' or 'text'")
        self.source, self.min_elsewhere = source, min_elsewhere
        self.path = str(sqlite_path)
        db = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True, check_same_thread=False)
        self.features: dict[str, set[str]] = {}
        self.versions: dict[str, set[str]] = {}
        for product, version, headings in db.execute("select product_id, product_version, headings from chunks"):
            self.versions.setdefault(product, set()).add(version)
            if source != "headings":
                continue
            for heading in json.loads(headings or "[]"):
                for gram in _feature_grams(_words(re.sub(r"^[\d.]+\s*", "", str(heading)))):
                    self.features.setdefault(product, set()).add(gram)
        self.db = db

    def _versions_with(self, product: str, phrase: str) -> dict[str, int]:
        """Chunk counts per version that contain the phrase (each word with or without a plural s)."""
        import itertools

        # words are plural-stripped ("analysis" -> "analysi"), so try each word with and without "s"
        forms = {" ".join(c) for c in itertools.product(*[(w, w + "s") for w in phrase.split()])}
        query = " OR ".join('"' + f.replace('"', "") + '"' for f in forms)
        rows = self.db.execute("select c.product_version, count(*) from chunk_fts f join chunks c on c.rowid=f.rowid"
                               " where chunk_fts match ? and c.product_id=? group by 1", (query, product))
        return {r[0]: r[1] for r in rows}

    def other_version_features(self, question: str, product: str | None, version: str | None) -> list[dict]:
        """Features the question names that the asked version's manual lacks but another version's has."""
        if not product or version in (None, "", "None") or len(self.versions.get(product, ())) < 2:
            return []
        version = str(version)
        mentioned = [v for v in self.versions[product] if v and re.search(rf"(?<![\d.]){re.escape(v)}(?![\d])", question)]
        if len(mentioned) > 1:
            return []  # a comparison across versions: absence in one of them is the point of the question
        grams = sorted(set(_feature_grams(_words(question))), key=lambda g: -len(g.split()))
        found, covered = [], []
        for gram in grams:
            if any(gram in c for c in covered):
                continue
            if self.source == "headings" and gram not in self.features.get(product, ()):
                continue
            counts = self._versions_with(product, gram)
            if not counts:
                continue  # in no manual at all: not a feature of this product (the name check's job)
            covered.append(gram)
            if version not in counts and (self.source == "headings" or max(counts.values()) >= self.min_elsewhere):
                found.append({"feature": gram, "versions": sorted(counts, key=_version_key)})
        return found


def _feature_grams(w: list[str]):
    """1-3 word phrases with content words at both ends; single words only when specific (7+ letters)."""
    names = set(PRODUCT_NAMES)
    for n in (3, 2, 1):
        for i in range(len(w) - n + 1):
            g = w[i:i + n]
            if g[0] in _FEATURE_STOP or g[-1] in _FEATURE_STOP or any(x.isdigit() for x in g):
                continue
            if any(x in names for x in g) and (n == 1 or any(x in _FEATURE_STOP for x in g)):
                continue  # "nextcloud office" is a feature; "symbol in kicad" and "qgis" are not
            if n == 1 and len(g[0]) < 7:
                continue
            yield " ".join(g)


def _version_key(v: str):
    return [int(x) if x.isdigit() else x for x in re.split(r"[.]", str(v))]


def version_reply(product: str, version: str, hit: dict) -> str:
    name = PRODUCT_NAMES.get(product, product)
    later = [v for v in hit["versions"] if _version_key(v) > _version_key(version)]
    where = (f"It is documented from {name} {later[0]} on" if later
             else f"It is documented in {name} " + ", ".join(hit["versions"]))
    return (f"The {name} {version} manual doesn't mention {hit['feature']}. {where}, so it may not be "
            f"available in {version}. Choose another version, or ask about what {version} offers instead.")


SUPPORT_PROMPT = (
    "You check answers written by a product-manual assistant. Look only at the numbered manual "
    "excerpts. The answer is SUPPORTED only if the excerpts state what it says for the product, "
    "version and feature the user asked about. If the answer gives steps, features, settings or "
    "names that the excerpts do not state, or answers a different feature than the one asked "
    "about, it is UNSUPPORTED. Reply with exactly one word: SUPPORTED or UNSUPPORTED. /no_think"
)


def support_messages(question: str, answer: str, evidence_text: str) -> list[dict]:
    return [{"role": "system", "content": SUPPORT_PROMPT},
            {"role": "user", "content": f"<manual_excerpts>\n{evidence_text}\n</manual_excerpts>\n\n"
                                        f"User question: {question}\n\nAnswer to check:\n{answer}"}]


def parse_support(text: str) -> bool:
    """Fail closed: only an explicit SUPPORTED (and no UNSUPPORTED) passes."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).upper()
    return "UNSUPPORTED" not in text and "SUPPORTED" in text
