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
| **Demand** — your customers | customer, segment, region, order, order line | `customers`, `sales_orders`, `sales_order_lines`, `ar_invoices`, `receipts` | **lit** — DB1 per customer is exact; revenue is Pareto-concentrated and accounts open and lapse, so concentration and prior-period metrics have answers |
| **Offer** — what you sell | product, product group, price list | `products` (standard_cost, list_price, validity window) | **lit** — DB1 per product group is exact; price realization derivable; a declared slice of the catalogue sells below contribution, so the portfolio can be pruned |
| **Capital** — where cash sits | receivable, payable, inventory position, WIP | AR + AP + `balance_sheet` + `stock_movements`, `inventory_positions` | **lit** — CCC = DIO + DSO − DPO is gradeable at both grains; no WIP until Throughput |
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

The registry then absorbed **FK topology** for the same reason. `metadata_truth`
re-listed the joins in a second place, and the operating chain had shipped without its
seven — `sales_orders.customer_id`, `sales_order_lines.order_id` and the rest were absent
from the published structural truth for as long as they were absent from the key maps. A
family now declares its joins where it declares its tables. Table-count assertions in the
tests read `default_tables()` too, so a new family no longer costs a round of magic-number
edits across five test files.

Then **shape**. `apply_normalization` named finance tables in its own function bodies —
`journal_lines LEFT JOIN journal_entries`, `chart_of_accounts` inlined into three facts —
so a new family's header/item pair would simply never collapse at `partial`, silently, and
visible only as a table count. A family now declares `Merge` (which parent/child pair
folds, on what key, renaming what) and `Fold` (which dimension inlines into which facts,
under which conformed concept). The transform executes the declarations and knows no table
names. `single`'s drop set is derived rather than listed — it was a literal tuple that had
already grown by six names, and a family missing from it left a table dangling beside the
mega-table.

Four things stopped being written twice in the process: `metadata_truth`'s merge-rename
map, its per-level table mappings, its folded-dimension block (authored, in its own words,
"to mirror `_inline_chart_of_accounts`" — a mirror is a second copy) and its
dimension→concept map. All four now read the declarations that perform the transform.
`tests/test_families_registry.py` runs a stand-in family through `apply_normalization` and
asserts it reshapes, which is the exit criterion exercised rather than asserted.

Still to come, and the reason this is only most of S0:

| Site | Hardcoding | State |
| :-- | :-- | :-- |
| `export.TABLE_NAMES` | a literal list of finance tables | **registry** |
| `schema_transforms` | `_KEY_COLUMNS`, `_NATURAL_KEYS`, `_LEGACY_NAMES` | **registry** |
| `metadata_truth._RELATIONSHIPS` | the FK topology re-listed away from the tables | **registry** |
| `export._write_manifest` | a literal version string, and run parameters repeated under `parameters` | **identity** (§6) |
| `schema_transforms` | the merge/inline functions name finance tables | **registry** |
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

### S1 · Inventory — completes Capital · **shipped**

*Shape reference: AdventureWorks `ProductInventory` + `TransactionHistory`.*

Shipped as specified below, with three decisions worth recording because they were not
obvious from the spec:

- **Movements are signed.** `units` and `value` are positive on a receipt and negative on
  an issue, so the roll-forward is a plain `SUM` rather than a case expression over
  `movement_type` — which is also what makes the movement table a genuine additive flow
  against the position's stock. The corpus's second stock/flow pair, and a sharper one
  than the balance tables: same key space, different tables, only meaning separating them.
- **Two locations, not one.** `location_id` over a single warehouse is a degenerate column
  that teaches nothing; the roll-forward and the GL tie both hold per location.
- **Receipts post before issues on the same day.** Ordering, not decoration: without it
  the fiscal year's first orders ship stock that has not arrived and on-hand goes negative
  on day one — a defect we would have invented ourselves.

The replenishment policy is a **designed coverage target**, not a forecast: each (product,
location) holds ~0.8–1.4 months of its own average demand, replenished in whole case-size
batches. Stated that way deliberately — a synthetic generator may use hindsight, but it
has to say so, or a consumer measuring "how good is this firm's planning" is measuring our
omniscience instead.

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

## 5. The oracle contract · **shipped**

`ground_truth.yaml` used to publish numbers without the definitions that produced them.
Grading a consumer against it was therefore manual: a defensible alternative definition
read as a delta to be argued rather than a variant to be matched.

**Every metric now publishes its definition and its legitimate variants, each with its own
values.** The registry is `src/testdata/oracle.py`; `ground_truth.metric_contract` binds
values to it and refuses to publish a metric without values or values without a
definition. 24 metrics, 5 variants.

```yaml
metrics:
  - id: dpo
    title: Days payable outstanding
    unit: days
    kind: ratio                 # stock over flow
    window: recompute           # ← derived from kind; a stock takes the window's LAST period
    basis: derived
    grains: [month, year]
    definition: "ap_balance[end of w] / purchases[w] * days[w]"
    scope: "Closing payable over the window's purchases; 0.0 when purchases are 0."
    values:
      month: {2025-01: 16.3, …}
      year:  {2025: 47.2}
    variants:
      - id: dpo_on_total_expenses
        definition: "ap_balance[end of w] / expenses[w] * days[w]"
        rationale: "the expense-denominator family; used where purchases are not separable"
        values: {…}
```

Four things the shape does that the sketch above did not:

- **`window` is published, not implied.** `flow`/`count` sums across a window, `stock`
  takes its last period, `ratio` is recomputed on the window's own aggregates. A consumer
  reading a quarter out of monthly values no longer has to infer which — averaging monthly
  ratios is the classic way a correct series produces a wrong number.
- **`values` is keyed by grain**, uniformly: `month`, `year`, `customer`, `product_group`.
  So a per-entity metric and a per-period one have the same shape, and `grains` is exactly
  the key set.
- **`basis`** is `derived` for everything today. The slot exists so a figure synthesized
  without a real basis — Throughput cost variance is the known case — is *marked* rather
  than footnoted.
- **The raw `annual` / `monthly` / `db1_by_*` blocks are gone from the file.** Every figure
  they held appears once, under a definition. Publishing both would restate every number
  without its definition, which is precisely how the defect below survived.

**It caught its first mislabel immediately.** `gross_profit` carried `revenue - total
expenses` — operating income. Nobody noticed because there was no definition beside it to
disagree with. Pinned now as `revenue - cogs`, with `operating_income`, `gross_margin` and
`operating_margin` published separately. §7's table already quoted a 26.6% "gross margin"
next to a "gross profit" that was 6.4% of revenue; the two lines were always different
things.

**Variants earn their keep, they are not cosmetic.** `operating_revenue` (accounts 41xx +
42xx, excluding 43xx other income) is the figure the order lines reconstruct to the cent —
so entity-grain revenue sums to the *variant*, never to the pinned metric. Without it the
gap reads as a reconciliation failure; with it, it is one named substitution. The same
applies to `cash_conversion_cycle_on_expense_dpo`: the cycle a consumer lands on when it
computes DPO the other way, published so the difference is not a mystery in a headline
number.

The rest of the contract:

- **Per-entity unit metrics** are first-class, not an appendix: `db1`, `db1_pct`,
  `units_sold`, `order_count`, `revenue` and `cogs` at customer and product-group grain
  today; DIO per product, OTIF per supplier, cost per capacity hour per asset as families
  land. A dimension without at least one graded per-entity metric is not lit.
- **Invariants** stay as they are (journal balanced, trial balance balanced, invoice↔payment
  matched, bank reconciliation rate) and grow per family (§4). The reconciliation rate is
  an *authored expectation*, never assumed to be 1.0 — a consumer reporting a perfect rate
  has overcleaned, and that is a failure.
- **Injection impact** may only report against a defined metric id or a declared
  `INTEGRITY_SURFACE` (`referential_integrity`, `benford_compliance`, …). Enforced at
  import: a target that looks like a metric id and has no definition behind it is the
  shape this whole section removes.
- **Provenance stamp** on every truth file — see §6.
- `metadata_truth.yaml` keeps carrying the structural layer — FK topology, table roles,
  semantic roles, stock vs flow per column, additivity, cycles, the conformed-dimension
  matrix. That is the floor a consumer's own structure detection is graded against, and
  it is a property of the data, not of any engine. Where the two files name the same
  metric, `tests/test_oracle.py` now pins the additivity verdict to the registry's `kind`
  — `metadata_truth` had described DPO as AP over COGS for a release after the pinned
  definition moved to purchases.

## 6. Corpus identity — reproducible, not frozen · **shipped**

`output/` is gitignored and stays that way. A corpus is not an artifact to preserve; it
is a **function of its parameters**, and the parameters are the contract:

```
(version, scenario, strategy, seed, months, fiscal start, normalization, family set, lever) → corpus
```

`CorpusIdentity` in `src/testdata/identity.py`; stamped as a `corpus:` block into
`manifest.yaml`, `ground_truth.yaml`, `metadata_truth.yaml`, `entropy_map.yaml`,
`sources.yaml` and `intervention.yaml`. A consumer pins the id, regenerates when it wants
the bytes, and can assert which corpus it graded against. Regeneration is the migration
path — freezing a directory is not.

Consequence to state plainly: **when a family lands, the same seed produces a different
corpus.** That is correct behaviour, not drift. It is also why the stamp matters — and S1
is the proof: the inventory family invalidated every previously generated directory, under
the same seed and the same version string, and nothing on disk said so.

Three additions to the tuple as originally written, all for the same reason — they change
the bytes:

- **Normalization.** `partial` drops three tables relative to `full`. Two directories that
  do not contain the same tables are not the same corpus.
- **Fiscal start.** Every date in the corpus hangs off it. It moved here out of
  `ground_truth.yaml`, which is now metrics only: seed, strategy and months were restated
  in that file's header beside a `generator: finance` that named the *vertical*, not the
  generator. `calculate_ground_truth` took `seed` and `strategy` as arguments it recorded
  and never read — provenance wearing the shape of a parameter.
- **The lever.** This is the sharp one. A levered run and its baseline are *defined* by
  differing; an identity that could not separate them would certify the wrong corpus in
  the one place where being wrong is worst. `intervention.yaml` therefore also carries
  `counterfactual_corpus_id` — the id of the same run without the lever — so "re-run
  without `lever`" is an instruction a consumer can check it followed.

**The digest is computed over exactly the published fields**, so a consumer can recompute
it rather than trust it. That is a constraint on what may enter the identity, not just a
courtesy: a hidden input would make the id unverifiable.

What the stamp deliberately does not capture is edits to `config/scenarios/*.yaml` and
`config/strategies/*.yaml` — the scenario and strategy *names* stand in for those, and the
version is the proxy for "the generator's own definitions moved". Which puts a standing
obligation on this repo: **bump the version whenever generation semantics change.** A stamp
that never moves is worse than no stamp, because it asserts a sameness nobody checked. The
version is read from package metadata, never restated in code — `export.py` used to carry
its own `"0.1.0"` literal beside `pyproject.toml`'s, the same one-fact-in-two-places shape
as the key maps. Landing this stamp bumped the version to **0.2.0**, which is the S1 corpus
change finally getting a name.

## 7. Known defects

**~~The inventory replenishment payable is never settled.~~ Fixed in S1.** The old
replenishment posted `DR Inventory / CR AP` once a month at 1.02–1.18 × the month's COGS
with no purchasing event behind it, so nothing ever cleared it. Every receipt is now a
vendor bill that ages and settles like any other. On `month-end-close / clean / seed 42 /
12 months`:

| | before | after |
| :-- | --: | --: |
| ending AP (fiscal-window) | 49,033,085.45 | 11,443,737.61 |
| … of which never debited back | 46,598,016.72 | 0.00 |
| annual DPO | 271.1 days | 59.2 days (purchases) / 62.4 (expenses) |
| annual DIO | — | 38.4 days |
| annual CCC | ungradeable | 71.4 days |

*Purchases* also became a computable quantity in the process — it needs goods bills to be
separable from expense bills, which is why the pinned DPO could not use it before.

**~~The expense base is sized independently of the firm.~~ Fixed in S1a.** The bottom line
read −3,660,147.05 against DB1 of ~+20.5M, because 24.2M of operating expense came from a
*fixed* 3,000 vendor invoices plus fixed monthly payroll and rent — counts that did not
move with the size of the business, so the P&L sign was an artifact of a knob rather than
a property of the firm. Operating expense is now sized off the **scale anchor**: the
contribution the order lines actually produce, times a declared
`opex_share_of_contribution`. The invoice count sets granularity, not spend.

| | `tiny` | `mid` |
| :-- | --: | --: |
| revenue | 28.26M | 178.06M |
| cost of sale | 20.74M | 122.86M |
| gross profit (rev − COGS) | +7.52M | +55.20M |
| gross margin | 26.6% | 31.0% |
| operating income (rev − all expenses) | **+1.82M** | **+13.68M** |
| operating margin | 6.4% | 7.7% |

The last two rows are the ones the defect moved. This table originally called the 1.82M
row "gross profit" while quoting a 26.6% gross margin above it — the same mislabelling
`ground_truth.yaml` carried, and the reason S1b pins definitions to numbers.

The anchor is computed at `lever=None` deliberately. A price or volume intervention must
not mechanically move payroll, or `intervention.yaml`'s "unaffected: the expenditure
cycle" becomes a false claim and the counterfactual stops being attributable.

**~~Master data is thin for the ladders it now carries.~~ Fixed in S1a.** 16 customers and
9 products could not support concentration, portfolio-tail or peer metrics. `mid` carries
400 customers and 240 products across 12 groups, with the shapes that make those metrics
answerable rather than merely computable — see §9.

**Free cash flow is a cold start, not a loss.** FCF is negative at every profile while
operating profit is positive, and the arithmetic is right: the corpus begins with zero
receivables and zero stock, so the first fiscal year absorbs a full AR balance (DSO ~86
days) and a full inventory position into working capital. That is a property of a corpus
that starts from nothing, not a defect — but a consumer benchmarking FCF against a going
concern should know which one it has.

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
| `tiny` | 16 | 9 / 4 | 20 | ~2.7k | fast tests; the shape the generator grew up with |
| `mid` | 400 | 240 / 12 | 120 | ~14.8k | the reference company — `month-end-close` asks for it |
| `large` | 4,000 | 1,200 / 24 | 600 | ~148k | stress, scan cost, tail behaviour |

`src/testdata/scale.py`; declared as `generator.scale_profile` in the scenario YAML and
stamped into the corpus identity (§6). The generator's own default stays `tiny`, so
calling it directly in a test suite is cheap.

Counts alone are not the point. **A population without a shape is not more realistic than
a handful** — 400 uniform customers make concentration risk unmeasurable exactly as 16 do.
Each profile therefore carries its distributions:

- **Revenue per customer is Pareto** — the top fifth of the book holds 41% of revenue at
  `tiny` and 53% at `mid`. Concentration risk has an answer instead of a shrug. The tail
  is capped (an uncapped Pareto occasionally draws a customer who *is* the firm) and
  renormalised by the **truncated** mean, so the shape changes concentration without
  quietly changing total volume.
- **Order value is log-normal** in units, so the mean and the median part company and a
  consumer reporting one for the other is visibly wrong.
- **A declared fraction of the catalogue sits below the contribution threshold** — priced
  thinly enough that the ordinary discount range drives realised unit contribution to zero
  or through it. A portfolio where every item earns cannot be pruned, and "what should we
  drop?" is a canonical Offer question that had no answer here.
- **Entities are born and die mid-year.** `customers.created_date` / `churned_date` and
  `products.launched_date` / `discontinued_date` are real columns, and they gate the
  *orders*, not just the master row — a validity window nothing respects is decoration.
  This is the case prior-period and peer comparisons actually fail on: a customer whose
  collapse is that they did not exist yet.

All of it is drawn from entity-keyed streams, so none of it disturbs a lever's
counterfactual.

## 10. Order of work

| | Work | Exit criterion |
| :-- | :-- | :-- |
| **S0** | The prune (§8) and the family registry (§3) | a family is added without editing export, schema transforms, ground truth and the runner |
| **S1** ✅ | Inventory + settle the replenishment payable | **met** — DPO 271 → 59.2 days; CCC gradeable monthly and annually; roll-forward and the GL tie hold per product, location and period |
| **S1½** ✅ | Corpus identity (§6) | **met** — one stamp across all six output files, digest recomputable from the published fields, levered run separable from its baseline |
| **S1a** ✅ | Scale profiles (§9) with shaped distributions, entity birth/death, and the expense base sized off the scale anchor | **met** — `tiny`/`mid`/`large` selectable and stamped; gross profit positive and plausible at every profile (§7); Pareto revenue, log-normal order value, a priced-thin catalogue tail and real validity windows |
| **S1b** ✅ | Oracle contract v2 (§5) | **met** — 24 metrics and 5 variants, each with its pinned definition, scope, window rule and per-grain values; publishing a metric without values or values without a definition raises; the raw metric blocks are gone, so every figure appears once |
| **S2** | Supply | OTIF, price variance, lead-time spread and effective cost per unit gradeable per supplier; three-way match exists with a declared exception rate |
| **S3** | Capacity | cost per capacity hour and utilization gradeable per asset against a declared ceiling; depreciation becomes asset-derived |
| **S4** | Throughput | yield, cycle-time efficiency and cost per step gradeable at team/line grain; the person-grain probe exists and no person column ships by default |
| **S5** | The typed lever set, on the CLI | one same-seed exact counterfactual per lever type, recorded in `intervention.yaml` |

Allocation (§4) becomes urgent at S3 and is a hard predecessor of any what-if.

## 11. Open decisions

1. **`metadata_truth.yaml` sections** — which structural sections earn their keep now that
   nothing external grades them. `relationships`, `table_roles`, `stock_flow` and
   `bus_matrix` clearly do; `metric_additivity` keyed by metric name is the one to review.
   S1b tightened rather than resolved it: where a name appears in both files the verdict
   is now pinned to the oracle's `kind`, but `metric_additivity` still carries entries for
   metrics this corpus does not compute (`ebitda`, `net_income`, `current_ratio`,
   `active_accounts`), which are engine vocabulary rather than corpus truth. Either they
   become computed metrics or the section shrinks to what the registry declares.
2. **The services archetype** as a second scenario (§4) — decide at S3.
3. **Allocation schemes** — which named keys ship first (freight by weight vs. by revenue
   vs. by order count is the canonical trio).

Decided:

- **Injection labels are consumer-agnostic** (§8) — the generator names the defect, the
  consumer maps it to its own machinery.
- **Scale is a profile** (§9), because generation cost is negligible and the metrics that
  need populations are canonical, not optional. `month-end-close` — the reference company
  — declares `mid`; the generator's own default stays `tiny` so a consumer's test suite
  is not made to pay for the reference corpus.
- **The scale anchor is computed at `lever=None`** (§7), so an intervention moves what it
  claims to move and nothing else.
