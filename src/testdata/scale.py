"""Scale profiles — how big the firm is, and what shape its populations have.

Generation is not the constraint; a 12-month corpus takes about a second. What is
constrained is a consumer loading CSVs in its own test suite, and the honesty of the
distributions. So scale is a **named profile** (§9), declared per scenario and stamped
into the corpus identity.

Counts alone are not the point. **A population without a shape is not more realistic
than a handful** — 400 uniform customers make concentration risk exactly as unmeasurable
as 16 do. Each profile therefore carries its distributions:

- **Revenue per customer is Pareto**, so "the top 5% of customers" is a real number and
  concentration risk has an answer rather than a shrug.
- **Order value is log-normal**, so the mean is not the median and a consumer that
  reports one for the other is wrong in a way we can see.
- **A tail of the catalogue sits below the contribution threshold** — products whose
  realised unit contribution is near zero or negative once discounting is applied. A
  portfolio where every product earns makes "what should we drop?" unanswerable.
- **Entities are born and die mid-year**, which is the case prior-period and peer
  comparisons actually fail on.

The last field is the one §7 was waiting for. ``opex_share_of_contribution`` sizes the
whole operating expense base off what the firm actually contributes, instead of a fixed
3,000 vendor invoices and a fixed monthly payroll that made the P&L sign an artifact of
a knob — implausibly loss-making at ``tiny`` and implausibly profitable at ``mid``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ScaleProfile:
    """One declared firm size, with the distributions that give it a shape."""

    name: str
    description: str

    # --- Populations ---
    customers: int
    products: int
    product_groups: int
    suppliers: int

    # --- Order intensity ---
    # Mean orders per customer per month BEFORE the customer's Pareto weight. The
    # realised annual order count is roughly customers x this x 12.
    orders_per_customer_month: float

    # --- Shape ---
    # Pareto tail index for per-customer order intensity. Lower = more concentrated;
    # 1.6 puts roughly half the revenue in the top fifth of the book.
    customer_pareto_alpha: float
    # The heavy tail is unbounded, and one customer drawing 60x would BE the firm.
    # Capping is a modelling choice, so it is declared rather than buried.
    customer_weight_cap: float
    # Units per order line, log-normal: median and sigma of the underlying normal.
    order_units_median: float
    order_units_sigma: float
    # Share of the catalogue priced below a defensible contribution threshold.
    tail_product_fraction: float
    # Share of customers and products that are born or die inside the fiscal year.
    churn_fraction: float

    # --- The expenditure cycle ---
    # Vendor bill count is sub-linear in firm size on purpose: a larger firm writes
    # larger invoices, not proportionally more of them.
    vendor_invoices: int
    # Operating expense as a share of contribution (revenue less cost of sale). At
    # 0.77 the firm earns an operating margin in the high single digits — a plausible
    # industrial company rather than one whose sign depends on a row count.
    opex_share_of_contribution: float


TINY = ScaleProfile(
    name="tiny",
    description="Fast tests and the shape the generator grew up with.",
    customers=16,
    products=9,
    product_groups=4,
    suppliers=20,
    orders_per_customer_month=18.0,
    customer_pareto_alpha=1.6,
    customer_weight_cap=4.0,
    order_units_median=9.0,
    order_units_sigma=1.0,
    tail_product_fraction=0.22,
    churn_fraction=0.15,
    vendor_invoices=3000,
    opex_share_of_contribution=0.77,
)

MID = ScaleProfile(
    name="mid",
    description="The reference company — populations large enough for concentration and tail metrics.",
    customers=400,
    products=240,
    product_groups=12,
    suppliers=120,
    orders_per_customer_month=2.9,
    customer_pareto_alpha=1.6,
    customer_weight_cap=8.0,
    order_units_median=9.0,
    order_units_sigma=1.0,
    tail_product_fraction=0.22,
    churn_fraction=0.15,
    vendor_invoices=9000,
    opex_share_of_contribution=0.77,
)

LARGE = ScaleProfile(
    name="large",
    description="Stress: scan cost, tail behaviour, and whether a method survives its own row count.",
    customers=4000,
    products=1200,
    product_groups=24,
    suppliers=600,
    orders_per_customer_month=2.9,
    customer_pareto_alpha=1.6,
    customer_weight_cap=12.0,
    order_units_median=9.0,
    order_units_sigma=1.0,
    tail_product_fraction=0.22,
    churn_fraction=0.15,
    vendor_invoices=30000,
    opex_share_of_contribution=0.77,
)

PROFILES: dict[str, ScaleProfile] = {p.name: p for p in (TINY, MID, LARGE)}

# ``tiny`` is the function-level default so that calling the generator directly stays
# cheap; scenarios declare what they actually want. §9 names ``mid`` the reference
# company, and ``month-end-close`` asks for it.
DEFAULT_PROFILE = "tiny"


def get_profile(name: str | ScaleProfile | None) -> ScaleProfile:
    """Resolve a profile by name, passing a profile object through unchanged."""
    if name is None:
        return PROFILES[DEFAULT_PROFILE]
    if isinstance(name, ScaleProfile):
        return name
    try:
        return PROFILES[name]
    except KeyError:
        raise ValueError(f"Unknown scale profile {name!r}. Available: {sorted(PROFILES)}") from None


# --- Name generation ---------------------------------------------------------
#
# Populations outgrew their hand-written lists. Names are composed from stems and
# suffixes by index, which is collision-free by construction and, being pure index
# arithmetic, does not consume a random draw — master data identity has to be stable
# across runs or every id in the corpus moves when a distribution is retuned.

_CUSTOMER_STEMS = [
    "Northwind Corp",
    "Contoso Ltd",
    "Adventure Works",
    "Fabrikam Inc",
    "Tailspin Toys",
    "Woodgrove Bank",
    "Litware Inc",
    "Proseware",
    "Alpine Ski House",
    "Trey Research",
    "Humongous Insurance",
    "Datum Corp",
    "A. Datum",
    "Coho Vineyard",
    "Lucerne Publishing",
    "Margie's Travel",
    "Blue Yonder",
    "Wide World Importers",
    "Consolidated Messenger",
    "Graphic Design Institute",
    "School of Fine Art",
    "City Power & Light",
    "Southridge Video",
    "Wingtip Toys",
    "Fourth Coffee",
    "Nod Publishers",
    "Lamna Healthcare",
    "Relecloud",
]
_CUSTOMER_SUFFIXES = ["", "Group", "Holdings", "Partners", "Industries", "International", "Services"]

_SUPPLIER_STEMS = [
    "Acme Corp",
    "Global Supply Co",
    "TechParts Inc",
    "Office Depot",
    "AWS",
    "CloudFlare",
    "Salesforce",
    "ADP Payroll",
    "Delta Airlines",
    "Marriott Hotels",
    "FedEx",
    "UPS",
    "Deloitte",
    "KPMG",
    "Ernst & Young",
    "PwC",
    "Google Workspace",
    "Microsoft",
    "Zoom",
    "Slack",
    "Rheinmetall",
    "Bosch Rexroth",
    "Festo",
    "Siemens Digital",
    "Norgren",
    "Parker Hannifin",
    "SKF",
    "Trelleborg",
    "Hoffmann",
    "Würth",
]
_SUPPLIER_SUFFIXES = ["", "Industries", "Systems", "GmbH", "Supply Co", "Partners", "Nordics"]


@dataclass(frozen=True)
class ProductGroupSpec:
    """One product group's vocabulary and its economics."""

    group: str
    families: tuple[str, ...]
    cost_low: float
    cost_high: float
    margin_low: float
    margin_high: float


# The first four groups are the ones the original nine-product catalogue used, in the
# order a four-group round robin reproduces it — a small profile keeps its shape.
_PRODUCT_GROUPS = (
    ProductGroupSpec(
        "Instruments",
        ("Flow Meter", "Pressure Sensor", "Thermal Probe", "Level Gauge", "Vibration Monitor"),
        95.0,
        460.0,
        0.38,
        0.46,
    ),
    ProductGroupSpec(
        "Controllers",
        ("Edge Controller", "PLC Module", "Servo Drive", "IO Block", "Gateway"),
        380.0,
        1400.0,
        0.29,
        0.37,
    ),
    ProductGroupSpec(
        "Consumables",
        ("Filter Cartridge", "Calibration Kit", "Seal Set", "Test Strip", "Lubricant Pack"),
        18.0,
        90.0,
        0.48,
        0.58,
    ),
    ProductGroupSpec(
        "Services",
        ("Installation Day", "Support Contract", "Commissioning", "Audit", "Training Day"),
        280.0,
        620.0,
        0.26,
        0.62,
    ),
    ProductGroupSpec(
        "Actuators", ("Ball Valve", "Linear Actuator", "Rotary Drive", "Damper", "Positioner"), 140.0, 900.0, 0.30, 0.41
    ),
    ProductGroupSpec(
        "Enclosures",
        ("Wall Cabinet", "Field Housing", "Junction Box", "Rack Frame", "Shield Plate"),
        60.0,
        520.0,
        0.33,
        0.44,
    ),
    ProductGroupSpec(
        "Connectivity",
        ("Bus Coupler", "Fieldbus Cable", "Wireless Bridge", "Patch Kit", "Antenna Set"),
        25.0,
        310.0,
        0.35,
        0.49,
    ),
    ProductGroupSpec(
        "Analytics",
        ("Condition Suite", "Reporting Module", "Forecast Pack", "Dashboard Seat", "API Tier"),
        90.0,
        700.0,
        0.55,
        0.74,
    ),
    ProductGroupSpec(
        "Safety", ("Light Curtain", "E-Stop Unit", "Interlock", "Guard Panel", "Relay Block"), 70.0, 640.0, 0.34, 0.45
    ),
    ProductGroupSpec(
        "Power", ("DIN Supply", "UPS Module", "Converter", "Busbar", "Surge Arrestor"), 45.0, 480.0, 0.31, 0.42
    ),
    ProductGroupSpec(
        "Handling", ("Conveyor Belt", "Roller Bed", "Gripper", "Transfer Arm", "Sorter Gate"), 210.0, 1600.0, 0.27, 0.36
    ),
    ProductGroupSpec(
        "Optics", ("Line Scanner", "Vision Head", "Lens Set", "Illuminator", "Filter Wheel"), 130.0, 1100.0, 0.36, 0.50
    ),
)

_VARIANTS = ("", "II", "Pro", "Compact", "XL", "HD", "Lite", "Plus", "Max", "S")

# Golden-ratio sequences: deterministic, evenly spread, and — unlike an RNG draw —
# free of draw-order coupling. Master data identity has to be stable, or retuning any
# distribution silently moves every id in the corpus.
_PHI = 0.6180339887498949
_PHI2 = 0.7548776662466927


def _spread(index: int, phase: float) -> float:
    """A deterministic quasi-random value in [0, 1) for *index*.

    Offset by one so index 0 does not always land on exactly 0.0 — which would make
    the first product of every catalogue the cheapest *and* always in the tail, a
    structure a consumer could read off the id instead of the data.
    """
    return ((index + 1) * phase) % 1.0


def _compose(index: int, stems: list[str], suffixes: list[str]) -> str:
    """A unique name for *index*, cycling suffixes and then numbering."""
    stem = stems[index % len(stems)]
    suffix = suffixes[(index // len(stems)) % len(suffixes)]
    cycle = index // (len(stems) * len(suffixes))
    name = f"{stem} {suffix}".strip()
    return name if cycle == 0 else f"{name} {cycle + 1}"


def customer_names(count: int) -> list[str]:
    """``count`` distinct customer names, stable for a given index."""
    return [_compose(i, _CUSTOMER_STEMS, _CUSTOMER_SUFFIXES) for i in range(count)]


def supplier_names(count: int) -> list[str]:
    """``count`` distinct supplier names, stable for a given index."""
    return [_compose(i, _SUPPLIER_STEMS, _SUPPLIER_SUFFIXES) for i in range(count)]


@dataclass(frozen=True)
class CatalogEntry:
    """One catalogue line: what it is, what it costs, and what it is meant to earn."""

    group: str
    name: str
    standard_cost: float
    margin_target: float
    # True when the item sits in the portfolio tail: priced so thinly that ordinary
    # discounting drives realised unit contribution to nothing or below.
    below_threshold: bool


def product_catalog(count: int, groups: int, tail_fraction: float) -> list[CatalogEntry]:
    """``count`` catalogue entries spread evenly over ``groups`` groups.

    Round-robin over groups rather than block-filling: a corpus where product ids sort
    by group makes the group a function of the id, which hands a consumer a structure
    it was supposed to discover. The tail is spread the same way, so "the losers" are
    not one group.
    """
    usable = _PRODUCT_GROUPS[: max(1, min(groups, len(_PRODUCT_GROUPS)))]
    out: list[CatalogEntry] = []
    for i in range(count):
        spec = usable[i % len(usable)]
        within = i // len(usable)
        family = spec.families[within % len(spec.families)]
        variant = _VARIANTS[(within // len(spec.families)) % len(_VARIANTS)]
        cycle = within // (len(spec.families) * len(_VARIANTS))
        name = f"{family} {variant}".strip()
        if cycle:
            name = f"{name} {cycle + 1}"

        # Cost log-spaced across the group's band — a catalogue is not uniform in price.
        u = _spread(i, _PHI)
        log_low, log_high = math.log(spec.cost_low), math.log(spec.cost_high)
        cost = math.exp(log_low + u * (log_high - log_low))

        tail = _spread(i, _PHI2) < tail_fraction
        if tail:
            # Thin enough that the discount range alone can take contribution negative.
            margin = 0.02 + 0.08 * _spread(i, _PHI)
        else:
            margin = spec.margin_low + _spread(i, _PHI2) * (spec.margin_high - spec.margin_low)
        out.append(CatalogEntry(spec.group, name, round(cost, 2), round(margin, 4), tail))
    return out
