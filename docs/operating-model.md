# The operating-model generator

The plan for growing this generator from a ledger into a full operating model, written
once so the design is not re-argued per family. It covers the six performance
dimensions, the framework that lets a dimension family plug in, the oracle contract a
consumer grades against, and the order of work.

Source of the dimension model: the retired `dataraum-eval` RFC set (RFC 0–6, 2026-07-27).
The framework around it is dead; the model stands and this repo is where it becomes
gradeable. **Nothing in this repo references a consumer.** Consumers bind to the corpus
and its truth files; the generator knows nothing about them.

---

## 1. Why this generator exists

Multi-table operating data with join topology, money attached and a time axis is exactly
the data nobody publishes — it carries a firm's competitive and personal-data exposure.
Public corpora can falsify a detector; they can never certify that a metric was computed
right, because nobody knows the true answer. The one thing we can build that does not
exist anywhere is **a multi-dimension operating corpus with an answer key**.

Two rules follow, and they are the whole design:

- **Money falls out of events, never bolted on as a column.** A sale is an order line at
  a quantity and a price; revenue, COGS, AR and the receipt are consequences. This is
  what makes the corpus internally consistent and a DB1 *true* rather than plausible.
- **Truth is designed with the entities, never after.** A table that lands without its
  truth fragment is not done.

A third rule keeps us honest about our own bias: **borrow documented real schema
shapes** (ERP/EAM/CRM export shapes) rather than inventing them. Borrowing governs the
provenance of the *shape*; generating the values is the point.

## 2. The six dimensions and where the generator stands

| Dimension | Canonical entities | Tables today | State |
| :-- | :-- | :-- | :-- |
| **Demand** — your customers | customer, segment, region, order, order line | `customers`, `sales_orders`, `sales_order_lines`, `ar_invoices`, `receipts` | **lit** — DB1 per customer is exact |
| **Offer** — what you sell | product, product group, price list | `products` (standard_cost, list_price) | **lit** — DB1 per product group is exact; price realization derivable |
| **Capital** — where cash sits | receivable, payable, inventory position, WIP | AR + AP + `balance_sheet`; GL account 1400 moves | **partial** — no stock ledger, and the payable inventory creates is never settled (§7, S1) |
| **Supply** — your suppliers | supplier, PO, PO line, goods receipt, claim | none — `invoices.vendor_id` is a bare string | **dark** |
| **Capacity** — what you run on | asset, site, line, shift, downtime, maintenance order | none | **dark** |
| **Throughput** — how work flows | work order, operation, step, scrap, team | none | **dark** |

Dimensions are **facets of one firm**, not verticals: one company, one calendar, one key
space. That is what makes the cross-dimension questions expressible at all — *which
customers are profitable on margin and unprofitable on cash* is a single query only
because Demand and Capital share a conformed customer key. Six corpora would make it
structurally inexpressible.

Where a family is absent the dimension stays **dark**. Darkness is information, and a
consumer's coverage map should read it off the corpus rather than be told.

## 3. What must change before family #2

Finance is hardwired in six places. Adding Supply on top of that shape means editing all
six again, and Capacity a third time.

**The cost is not hypothetical — it already bit.** The operating chain reached
`export.TABLE_NAMES` but never `_KEY_COLUMNS`, `_NATURAL_KEYS` or `_LEGACY_NAMES`, so
`customer_id`, `product_id`, `order_line_id`, `ar_invoice_id` and `receipt_id` were
silently skipped by every key strategy and left unspelled in a legacy export. One of four
maps was updated and nothing said so.

`src/testdata/families.py` now holds the declarations — tables, primary keys, natural-key
prefixes, legacy spellings — and the exporter and both schema transforms read from it.
`tests/test_families_registry.py` fails if a table exists on the dataset without a
declaration, so the next family cannot repeat the omission. It also surfaced a genuine
ambiguity: `order_id` is claimed by both the sales order and the role-play probe fact over
unrelated id spaces, so key strategies now leave it alone rather than fuse two populations
into one.

Still to come, and the reason this is only half of S0:

| Site | Hardcoding | State |
| :-- | :-- | :-- |
| `export.TABLE_NAMES` | a literal list of finance tables | **registry** |
| `schema_transforms` | `_KEY_COLUMNS`, `_NATURAL_KEYS`, `_LEGACY_NAMES` | **registry** |
| `schema_transforms` | the merge/inline functions name finance tables | open |
| `ground_truth.GroundTruth` | finance-specific fields (`ar_balance`, `dso`, …) | open |
| `metadata_truth` | `VERTICAL = "finance"`, one canonical authored blob | open |
| `scenarios/runner` | imports `generate_finance_dataset` directly | open |
| `FinanceDataset` | one fixed container of eight-plus lists | open |

**Proposal — a family registry.** A *family* is a cohesive set of tables with its own
generator, GL postings, truth fragment and schema metadata:

```python
@dataclass(frozen=True)
class Family:
    name: str                                   # "core_ledger", "operating_chain", "inventory", …
    tables: tuple[str, ...]
    requires: tuple[str, ...]                   # families it generates against
    generate: Callable[[GenContext], FamilyOutput]   # rows + GL postings + lever hooks
    truth: Callable[[Corpus], TruthFragment]         # metrics, per-entity values, invariants
    key_columns: Mapping[str, str]              # column -> owning table
    legacy_names: Mapping[str, str]
```

- `GenContext` carries seed, fiscal calendar, ID counters, the master data already
  generated, and the **entity-keyed RNG** (`_stream(seed, purpose, entity_id)`) — never
  the sequential stream, or a volume lever silently re-rolls unrelated draws.
- Families emit GL postings rather than writing the ledger themselves; the runner
  assembles one ledger, so the closed-loop invariant survives every new family by
  construction.
- `FinanceDataset` becomes `Corpus`: `tables: dict[str, list[BaseModel]]` plus the family
  manifest that produced them.
- The scenario YAML gains `families: [core_ledger, operating_chain, inventory]`. The
  existing `tables:` key is display-only today and is superseded by it.
- `ground_truth` composes per-family fragments against one metric registry (§5); a
  dimension appears in the scorecard only when its family is active.

This is the only structural work that is not a family. It should land before Supply.

## 4. The family specs

The durable part of this document. Each family names its shape reference, its tables, the
GL it posts, the metrics it lights, the truth it must export, and the levers it enables.

### S1 · Inventory — completes Capital

*Shape reference: AdventureWorks `ProductInventory` + `TransactionHistory`.*

- `stock_movements` — `movement_id`, `product_id`, `location_id`, `date`,
  `movement_type` (receipt | issue | adjustment), `units`, `unit_cost`, `value`, and the
  document it came from (order line, replenishment, count).
- `inventory_positions` — a **stock** at `(product_id, location_id, period)`:
  `units_on_hand`, `unit_cost`, `value`. Never summed across periods.
- GL: receipt `DR 1400 / CR AP`, issue `DR 5100 / CR 1400`, adjustment to a shrinkage
  expense. The COGS leg already posts this way; the movement table becomes its subledger.
- Fixes the S1 defect (§7): the payables inventory creates get settled.
- Metrics: DIO by product and product group, **CCC = DIO + DSO − DPO** — the anchor
  metric, gradeable for the first time; inventory carrying cost at a named, versioned rate.
- New invariant: `opening + receipts − issues ± adjustments = closing`, per product and
  period, and the position table reconciles to GL account 1400.
- Levers: timing (reorder point, batch size), volume.

### S2 · Supply

*Shape reference: AdventureWorks `Purchasing` (Vendor, PurchaseOrderHeader/Detail, with
`RejectedQty` as the real quality signal).*

- `suppliers` — `supplier_id`, `name`, `supplier_group`, `country`, `payment_terms`,
  `promised_lead_time_days`. Turns today's `invoices.vendor_id` string into a real FK.
- `purchase_orders` / `purchase_order_lines` — `product_id`, `units`, `unit_price`,
  `promise_date`.
- `goods_receipts` — `po_line_id`, `receipt_date`, `units_received`, `units_rejected`.
  The receipt↔PO-line join is the link real corpora are always missing; here it exists,
  which is what makes OTIF computable.
- `supplier_invoices` — today's `invoices`, extended with `po_id`, so the **three-way
  match** (PO ↔ receipt ↔ invoice) is a real structure with a *designed* exception rate.
- `claims` — `supplier_id`, `receipt_id`, `date`, `amount`, `reason_code`.
- GL: receipts feed `stock_movements` (S1's receipt leg), invoices post AP, claims credit
  cost of sale.
- Metrics: OTIF, price variance per line, lead-time **spread** (not the mean — the mean is
  a vanity metric), claim-to-spend, effective cost per unit delivered (price + delay +
  claim), DPO per supplier.
- Truth: per-supplier OTIF, price variance, effective cost; the three-way-match exception
  rate as a declared parameter, not an accident.
- Levers: price, timing (lead time), rate (defect rate), mix (dual sourcing).

### S3 · Capacity

*Shape reference: EAM export shapes (SAP PM `Equipment` / `MaintenanceOrder` /
`MeasurementDocument`; Maximo `Asset` / `Meter`). No open ERP schema covers this well —
designing it ourselves is legitimate provided the shape is checked against a real system
or a wild corpus in the same family.*

- `locations` — site → line → work centre, a real hierarchy (also the Capacity ladder).
- `assets` — `asset_id`, `location_id`, `asset_class`, `acquisition_date`,
  `replacement_value`, `capacity_uom`, `theoretical_hours_per_period`. The ceiling is an
  explicit, versioned number: most utilization disputes are about the denominator.
- `shift_calendar` — `location_id`, `date`, `shift`, `planned_hours`.
- `downtime_events` — `asset_id`, `start`, `end`, `hours`, `reason_code`, `planned` flag.
- `maintenance_orders` — `asset_id`, `type` (preventive | corrective), `labour_cost`,
  `parts_cost`, `downtime_hours`.
- `meter_readings` — `asset_id`, `date`, `meter`, `value`.
- GL: maintenance cost to expense with the line as cost centre; **depreciation becomes
  asset-derived** rather than the current random monthly draw — a strict improvement to
  the existing ledger.
- Metrics: cost per capacity hour, utilization against the stated ceiling, output per
  capacity unit, MTBF, maintenance cost index.
- Target note: the theoretical ceiling is a *definitional* target, typed distinctly so it
  is never confused with a plan.
- Levers: capacity (add a shift, add a line), rate (uptime), timing (maintenance interval).

### S4 · Throughput

*Shape reference: AdventureWorks `WorkOrder` / `WorkOrderRouting` for the shape — with the
known caveat that its `ActualCost == PlannedCost` on every routing row, so **cost variance
is the one figure we synthesize**, and the corpus must say so.*

- `work_orders` — `product_id`, `location_id`, planned/actual start and end, `planned_qty`,
  `good_qty`, `scrap_qty`, `status`.
- `operations` — `wo_id`, `step_no`, `work_centre_id`, `standard_minutes`,
  `actual_minutes`, `team_id`.
- `scrap_events` — `operation_id`, `reason_code`, `units`, `cost`.
- `teams` — `team_id`, `shift`, `location_id`. **Aggregate entities only.**
- GL: production variance to a variance account; scrap cost to expense.
- Metrics: cost per process step, cycle-time efficiency (standard ÷ actual), first-pass
  yield, rework cost per unit.
- **Person grain is a design rule, not a degraded mode.** Canonical entities are shift,
  line, team, work centre, work order — every metric an executive acts on, none a works
  council objects to (§87 BetrVG, GDPR). The corpus ships no person column by default.
  It *should* ship an opt-in probe table carrying a `worker_id`, so a consumer can prove
  its own refusal fires — a quarantine rule nobody can test is not a rule.
- Levers: rate (yield, cycle time), capacity (shift structure), timing (batch size).

### S5 · Levers as DGP parameters

One lever type exists (`price_level`, Python-API only). The pattern is right and it
generalizes: a lever is a **DGP parameter**, applied after every RNG draw with no control
flow branching on values, so a same-seed pair is an *exact* counterfactual, recorded in
`intervention.yaml`.

Complete the typed set — volume, price, mix, rate, timing, capacity, allocation key — one
per dimension family as it lands, and expose them on the CLI. The exactness constraint is
already met by entity-keyed RNG streams; a volume lever changes event *counts*, which
would diverge a sequential stream.

### Cross-cutting · Allocation

"Cost per customer" is an allocation, not an aggregation, and different keys yield
different truths. Ship it as **named, plural schemes computed side by side** — *profitable
under key A, not under key B* is the point, not an edge case. DB1 (direct costs, no
allocation) is what we have and is shippable alone; DB2/DB3 need the allocation object and
it becomes urgent when Capacity lands, because shared-capacity cost is Offer's input. One
scheme must span dimensions, so the object is global, not per-family.

### Archetype note

S3 + S4 make this firm asset-intensive manufacturing. One firm cannot also be
contract-intensive, so a services archetype (grade / role-hour pool / project, with
utilization × realization) is a **second scenario** reusing Demand, Offer and Capital with
a different Capacity/Throughput pair — not a second repo and not a second truth format.
Decide when S3 lands, not before.

## 5. The oracle contract

Today `ground_truth.yaml` publishes numbers without the definitions that produced them.
Grading a consumer against it is therefore manual: a defensible alternative definition
reads as a delta to be argued rather than a variant to be matched.

**Every metric publishes its definition and its legitimate variants, each with its own
value.** Grading becomes mechanical, and the "which definition do we pin?" conversation
happens once, here, instead of after every run.

```yaml
metrics:
  - id: dpo
    title: Days Payable Outstanding
    unit: days
    kind: ratio                 # stock over flow — a window takes the stock's LAST period
    grains: [month, year]
    definition: "ap_balance[end of w] / purchases[w] * days[w]"
    scope: "AP accounts 2110,2120; purchases = vendor-bill credits to AP"
    values: {2025-01: 16.3, …, annual: 48.5}
    variants:
      - id: dpo_on_total_expenses
        definition: "ap_balance[end of w] / total_expenses[w] * days[w]"
        rationale: "the expense-denominator family; used where purchases are not separable"
        values: {…}
```

The rest of the contract:

- **Per-entity unit metrics** are first-class, not an appendix: DB1 per customer and per
  product group today; DIO per product, OTIF per supplier, cost per capacity hour per
  asset as families land. A dimension without at least one graded per-entity metric is
  not lit.
- **Invariants** stay as they are (journal balanced, trial balance balanced, invoice↔payment
  matched, bank reconciliation rate) and grow per family (§4). The reconciliation rate is
  an *authored expectation*, never assumed to be 1.0 — a consumer reporting a perfect rate
  has overcleaned, and that is a failure.
- **Provenance stamp** on every truth file: generator version, scenario, strategy, seed,
  months, family set. A figure that must be synthesized without a real basis (Throughput
  cost variance is the known case) is marked as such rather than footnoted.
- `metadata_truth.yaml` keeps carrying the structural layer — FK topology, table roles,
  semantic roles, stock vs flow per column, additivity, cycles, the conformed-dimension
  matrix. That is the floor a consumer's own structure detection is graded against, and
  it is a property of the data, not of any engine.

## 6. Corpus identity — reproducible, not frozen

`output/` is gitignored and stays that way. A corpus is not an artifact to preserve; it
is a **function of its parameters**, and the parameters are the contract:

```
(generator version, scenario, strategy, seed, months, family set) → corpus
```

That tuple is stamped into `manifest.yaml` and into every truth file. A consumer pins the
identity string, regenerates when it wants the bytes, and can assert which corpus it
graded against. Regeneration is the migration path — freezing a directory is not.

Consequence to state plainly: **when a family lands, the same seed produces a different
corpus.** That is correct behaviour, not drift. It is also why the stamp matters — a
consumer holding a stale directory must be able to detect that, and today it cannot.

## 7. Known defects

**The inventory replenishment payable is never settled.** `_generate_inventory_replenishment`
posts `DR Inventory / CR AP` once a month at 1.02–1.18 × the month's COGS, and no payment
cycle ever clears it. Measured on `month-end-close / clean / seed 42 / 12 months`:

| | value |
| :-- | --: |
| ending AP (GL net) | 49,033,085.45 |
| … of which unsettled replenishment credits (12 entries) | 46,598,016.72 |
| … ever debited back | 0.00 |
| annual DPO | 271.1 days |
| gross profit (revenue − all expenses) | −3,635,249.05 |

DB1 per product group is simultaneously ~+20.5M, so the corpus states that the firm has
healthy unit economics and a negative P&L, with 95% of its payables permanently open. The
inventory *asset* side is sound (closing 3.83M against COGS 42.77M ≈ 33 days). Only the
credit leg dangles.

This is S1's first fix, and it is why S1 is first: the honest way to settle those payables
is the supplier side, which is Supply's first vertebra.

**Master data is thin for the ladders it now carries.** 16 customers and 9 products across
4 product groups will not support customer-concentration, portfolio-tail or peer-comparison
metrics, all of which are canonical for Demand and Offer. Raise before those metrics are
declared gradeable — §9 sizes it.

## 8. The prune

`dataraum-context` and `dataraum-eval` are retired. What was authored for them goes; what
is a property of the data stays.

| What | Why | Replacement | State |
| :-- | :-- | :-- | :-- |
| `.claude/handoff.md` | correspondence with the retired repos | — | **done** |
| `detector_id` on every injection (`type_fidelity`, `null_semantics`, `benford`, …) | names another system's detectors | `defect` + `defect_detail` in a closed, generator-owned vocabulary, beside the existing `layer`. A strategy may set `consumer_hint:` to carry its own label through; the generator never reads it | **done** |
| Ticket and ADR references throughout (132 of them) | a retired tracker | the empirical fact, without the system that recorded it | **done** |
| Engine constants quoted as justification (`REF_UNIQUENESS_MIN`, `FIRE_RESIDUAL_MAX`, `phases/typing.yaml` `min_confidence`) | those thresholds were another system's | keep the parameter and the number, restate the reason generically — a representative threshold a consumer applies | **done** |
| `metadata_truth` framed as "the agent layer of the engine's `current_*` views" | binds truth to a dead engine's vocabulary | framed as structural truth about the data: topology, roles, stock/flow, additivity, cycles, conformance | **done** |
| `VERTICAL = "finance"` and the six hardcodings in §3 | one hardcoded vertical | the family manifest (§3) | S0 |

The defect vocabulary, closed: `type_fidelity`, `completeness`, `null_encoding`,
`distribution`, `temporal_stability`, `unit_consistency`, `business_meaning`,
`measure_behavior`, `format_consistency`, `referential_integrity`, `relationships`,
`join_paths`, `dimensional_structure`, `cross_table_consistency`, `derived_consistency`,
`driver_effect`. A new injector picks from this set or the set grows deliberately.

Kept deliberately: the **probe tables** (`addresses` / `orders` / `deliveries` for
role-playing FKs, `ref_entities` / `ref_activity` for labelled relationship pairs,
`measure_probes`, `formula_probes`). They carry labelled ground truth for structure
detection, they cost nothing when no strategy injects into them, and that class of
detector did not die with the framework. Kept too: the **injection family framework** —
sampling a corruption from a parameter space beats a fixed fixture, because a detector
that memorized one token proves nothing.

## 9. Scale

Generation is not the constraint. A full 12-month clean corpus — 35k journal lines across
15 tables — takes **0.86 s**. Ten times the master data is ten seconds, and the machine
does not care. What *is* constrained: a consumer loading CSVs in its own test suite, and
the honesty of the distributions. So scale is a **named profile**, declared per scenario:

| Profile | Customers | Products / groups | Suppliers | Orders/yr | Use |
| :-- | --: | --: | --: | --: | :-- |
| `tiny` | 16 | 9 / 4 | 12 | ~3.6k | fast tests, the current shape |
| `mid` *(default)* | 400 | 240 / 12 | 120 | ~14k | the reference company |
| `large` | 4,000 | 1,200 / 24 | 600 | ~140k | stress, scan cost, tail behaviour |

Counts alone are not the point. **A population without a shape is not more realistic than
a handful** — 400 uniform customers make concentration risk unmeasurable exactly as 16 do.
Each profile therefore carries its distributions: revenue per customer Pareto (so the top
5% is a real number), order value log-normal, a portfolio tail of products below
contribution threshold, and **entity birth and death** — customers and products that
appear and disappear mid-year, which is what makes prior-period and peer targets face the
case they actually fail on.

## 10. Order of work

| | Work | Exit criterion |
| :-- | :-- | :-- |
| **S0** | The prune (§8) and the family registry (§3) | a family is added without editing export, schema transforms, ground truth and the runner |
| **S1** | Inventory + settle the replenishment payable | DPO returns to a sane band; CCC = DIO + DSO − DPO gradeable monthly and annually; the roll-forward invariant holds |
| **S1b** | Oracle contract v2 (§5) | every metric carries its definition and variants; a consumer's alternative grades as a named variant, not as an unexplained delta |
| **S2** | Supply | OTIF, price variance, lead-time spread and effective cost per unit gradeable per supplier; three-way match exists with a declared exception rate |
| **S3** | Capacity | cost per capacity hour and utilization gradeable per asset against a declared ceiling; depreciation becomes asset-derived |
| **S4** | Throughput | yield, cycle-time efficiency and cost per step gradeable at team/line grain; the person-grain probe exists and no person column ships by default |
| **S5** | The typed lever set, on the CLI | one same-seed exact counterfactual per lever type, recorded in `intervention.yaml` |

Allocation (§4) becomes urgent at S3 and is a hard predecessor of any what-if.

## 11. Open decisions

1. **`metadata_truth.yaml` sections** — which structural sections earn their keep now that
   nothing external grades them. `relationships`, `table_roles`, `stock_flow` and
   `bus_matrix` clearly do; `metric_additivity` keyed by metric name is the one to review.
2. **The services archetype** as a second scenario (§4) — decide at S3.
3. **Allocation schemes** — which named keys ship first (freight by weight vs. by revenue
   vs. by order count is the canonical trio).

Decided:

- **Injection labels are consumer-agnostic** (§8) — the generator names the defect, the
  consumer maps it to its own machinery.
- **Scale is a profile, `mid` by default** (§9), because generation cost is negligible and
  the metrics that need populations are canonical, not optional.
