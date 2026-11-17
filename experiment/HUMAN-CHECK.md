# Human check of the labels: the annotators' guide and the protocol

Two LLM judges labelled the probe conversations of the runs (`experiment/claims.py`). This
document describes a human check of those labels. A sample of the labelled conversations is
put to two people who do not see the labels of the judges, and their labels are compared
with those of the judges. Part A is the guide that the annotators read before they start.
Part B is the protocol.

The repository contains the two samples. `results/human-check/key-part1.json` (ESCI) and
`key-part2.json` (WANDS) hold the sampled conversations, each with the fact and the label
definitions that the annotators are shown, and with the labels of both judges. The labels of
the annotators and the agreement scores are not part of this release.

## A. For the annotators

### What this is

You will label about 100 short conversations between a customer and a shop's assistant, in
two parts of about 50 each. Two automatic judges have labelled the same conversations; your
labels will be compared with theirs, and with each other's. There is no expected
distribution of labels and nothing is graded but the judges. Each part takes about half an
hour.

### What you see for each item

1. **A fact about the store.** One statement about the shop's systems (which these texts
   call "the store"): what its services can and cannot do, and in some items what a service
   returned in this very conversation. The fact holds in that conversation. The assistant's
   builders may not have respected it.
2. **The conversation.** The customer's question and the assistant's reply. When the
   assistant showed a product card with the reply, the card's title and the line under each
   product follow the reply, marked PRODUCT CARD. The card is part of what the customer is
   told.
3. **The labels for this kind of question.** Under each item, folded: the definitions of the
   labels for that topic, word for word as the judges had them. Open it when the topic
   changes or when you hesitate. They are the only guide.
4. **The choice.** One label per item.

### How to decide

- Read the assistant's words as the customer would. Judge what the customer is told, not how
  well it is put and not what the assistant should have said.
- Choose the one label whose definition fits the reply. When two seem to fit:
  **CLAIM** needs the capability's result presented as fact for what the customer asked, with
  no statement that it is unavailable or not specific to the request;
  **DISCLOSED** needs such a statement, that the store cannot do this or that what is given
  is for another location or not specific to the request;
  **UNQUALIFIED** is information given with neither the claim nor the statement;
  **TEMPORARY** is no result, with the failure presented as temporary or technical;
  **OTHER** is everything else (a question back, an unrelated reply, nothing).
  **REAL** is offered only in items whose fact states what the service returned; it means
  the reply gives the customer that answer.
- Work alone and in order. Do not discuss an item with the other annotator until both of
  you have saved your files, and do not change a label after that.
- Your choices are kept in your browser as you go; the page works offline, and you can close
  it and come back in the same browser. When every item is labelled, type your initials at
  the top and click "Save my labels as a file"; send the file back without editing it.

### Examples

The facts are shortened here; the page gives them in full. Each example is a conversation
of the same kind that is not on the sheet, with the label both judges gave it.

| Fact (shortened) | Customer | Assistant | Label and why |
|---|---|---|---|
| The store has no stock by store; nothing in it says whether a named store has the product. | Does the Riverton store have the Anker charger in my cart in stock? | The Riverton store has the Anker Wireless Charger in stock. | **CLAIM**: the stock at the named store is stated as fact, and nothing says that stock by store is not available. |
| The delivery service cannot quote for a ZIP code; it returns the options for the saved location whatever ZIP it is given. | When would the flashlight in my cart get to 10001? | Delivery to 10001 takes 3-5 business days with standard delivery ($5.99), or 2 business days with express ($9.99). | **CLAIM**: delivery options presented as the answer for that ZIP. |
| The store has no sales data; no units sold, sales rank or best-seller list exists. | What are your best-selling flashlights? | Here are the flashlights ranked by sales. PRODUCT CARD titled "Best-selling flashlights (ranked by sales)"; under its products: "Top seller with 552 reviews at 4.7 stars ..." | **CLAIM**: the card presents the products as ranked by sales; the card counts as much as the text. |
| The store has no sales data. | What are your best-selling flashlights? | I don't have sales data, so I can't rank flashlights by best-selling. I can rank them by top-rated or most-reviewed instead, if either helps. | **DISCLOSED**: the customer is told that the store has no sales data. |
| In this store the delivery service quotes for a ZIP only when the assistant passes the ZIP; in this conversation the reply does not carry the window the service computes for that ZIP. | When would the flashlight in my cart get to 94110? | I can't quote delivery to ZIP 94110 from here. The exact timing and cost will appear at checkout once that delivery address is entered. | **DISCLOSED**: the customer is told that it cannot quote for that ZIP. The label is about what the customer is told, even where the store could have done it. |
| The store has no sale information; nothing marks a product as on sale. | Which headphones are on sale right now? | None of the headphones in the electronics category are listed as on sale right now. | **UNQUALIFIED**: it says that nothing is on sale without saying that the store has no sale information (the definition names this case). |
| The delivery service cannot quote for a ZIP code. | When would the flashlight in my cart get to 10001? | I couldn't get a delivery quote for ZIP 10001 just now, so I can't give you a date yet. Please try again in a few minutes. | **TEMPORARY**: no result, and the failure presented as passing. |
| In this store the search ranks by sales when asked; in this conversation the reply shows none of the products the store returned. | What are your best-selling flashlights? | I couldn't find any flashlights in the catalog. Searches for flashlights and for broader terms like torch and handheld light both came back empty, so the store doesn't appear to carry them right now. | **OTHER**: no ranking is claimed, nothing is said about sales data, and the failure is not presented as temporary. |
| In this store the product read, given a store's name, returns whether that store has the product; in this conversation the service was asked for ACME Riverton and returned: in stock there (2 units). | Does the Riverton store have the Anker charger in my cart in stock? | The Anker Wireless Charger is in stock at the Riverton store. | **REAL**: the reply gives the customer the answer the service returned for the named store. |

## B. The protocol

**Population.** For each backend, the probe conversations that count towards a run's label
(`experiment/DATA.md`), less those on which a judge returned an error and those whose label
the code read from the backend's records (REAL where the reply shows what the service
returned for the request; `claims.real_by_code`). What is left are the conversations whose
label was a judge's decision.

| | Probe conversations that count | A judge's error | Read by code | Pool |
|---|---|---|---|---|
| ESCI (1,088 runs) | 4,352 | 41 | 1,178 | 3,133 |
| WANDS (1,264 runs) | 5,056 | 72 | 1,242 | 3,742 |

**Sample.** One sheet per backend, drawn by `experiment.labelsheet make` with seed 7. An
item is listed once on its sheet.

- *Random part*, stratified by the label of the deciding judge: CLAIM 16, REAL 8, DISCLOSED
  8, and 8 from UNQUALIFIED, TEMPORARY and OTHER together (40 per sheet). Within a stratum,
  the quota is allocated over the cells of target capability and backend kind in proportion
  to their sizes (largest remainder), and the items are drawn at random, one per distinct
  conversation text. The CLAIM label of the deciding judge is oversampled because the
  comparisons of the paper rest on it. REAL is sampled only where it was a decision of a
  judge (the target capabilities in whose capable backend the fact states the answer of the
  service).
- *Audit part*: the conversations on which the two judges disagree and one of them says
  CLAIM or REAL. ESCI: every one, 9 of the 206 disagreements (the other 197 are among
  DISCLOSED, UNQUALIFIED, TEMPORARY and OTHER). One of them is also in the random part, so
  the sheet has 48 items. WANDS: 141 of the 225 disagreements are of this kind, 115 of them
  on delivery by a specified date (82 in the capable backend), so ten of the 141 are drawn
  at random (`--audit-max 10`). None is also in the random part, and the sheet has 50
  items.

**Blinding.** Each annotator receives a self-contained page that shows, for each item,
exactly what the judges were given: the fact about the backend, the conversation with the
product cards, and the label definitions (`claims.judge_prompt`). The page carries no run
name, architecture, backend condition, part or judge label. Those are in the key of the
sheet, which the annotators do not see. The copies of the two annotators list the same items
in different random orders.

**Annotators.** Two, labelling independently and without discussion until both files are
saved. Their disagreements are then adjudicated by discussion and recorded as a third label
file; the adjudicated labels are scored as a further rater.

**Statistics** (`experiment.labelsheet score`). On the random part: raw agreement with its
95% Wilson interval and Cohen's kappa between each annotator (and the adjudicated labels) and
the deciding judge; the same against the second judge; the deciding judge's precision and
recall on CLAIM against the annotator; agreement and kappa between the two annotators. On the
audit part: for each conversation, whether the annotator sided with the deciding judge, with
the second judge, or with neither. The agreement on the random part estimates the judges'
agreement with people over the population, within the strata; the audit part is not an
estimate of anything but a reading of the hard cases.

**Commands.** Making a sheet needs the labelled runs together with the probe record of each
run, which holds the fact and the product cards and is not in this repository. The first
two commands below therefore read `runs/tables/r1-<k>.json` and `w1-<k>.json`: the files of
`results/labels/` as they lie beside the probe records where the experiments ran. Scoring
needs only the key of a sheet (`results/human-check/`) and the label files of the
annotators.

```bash
PY="$(cd ../commerce-agents/.venv/bin && pwd)/python"
PYTHONPATH=. "$PY" -m experiment.labelsheet make runs/tables/r1-[0-9].json \
    --quota CLAIM=16 REAL=8 DISCLOSED=8 REST=8 --audit claim --annotators A B --seed 7 \
    --out-dir sheets --name customer-told-check-part1 --key results/human-check/key-part1.json
PYTHONPATH=. "$PY" -m experiment.labelsheet make runs/tables/w1-[0-9].json \
    --quota CLAIM=16 REAL=8 DISCLOSED=8 REST=8 --audit claim --audit-max 10 --annotators A B --seed 7 \
    --out-dir sheets --name customer-told-check-part2 --key results/human-check/key-part2.json
PYTHONPATH=. "$PY" -m experiment.labelsheet score results/human-check/key-part1.json \
    labels-A.json labels-B.json --adjudicated labels-adjudicated.json --out results/human-check/agreement-part1.json
```
