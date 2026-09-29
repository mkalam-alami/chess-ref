# Annotating starting-position photos (Wave 2)

This is the workflow for turning photos dropped into this directory into ground truth for the real-photo bench
(`BENCH_ONLY=realocc npm run bench`, see `tests/synth/realOccBench.ts`). The tools are in `tests/tools/annotate.ts`,
and you run them with `npm run annot`. They take their options from environment variables, listed at the top of
`tests/tools/annotate.entry.ts`. All PNG output goes to `tests/tools/annot-out/<dir name>/`, which is gitignored. Open
those PNGs with the Read tool.

## Ground truth format

`corners.json` uses the same format as `tests/fixtures/real/corners.json`:

```json
"03-green-vinyl-topdown.jpg": { "width": 1600, "height": 1200, "corners": [[x0,y0],[x1,y1],[x2,y2],[x3,y3]], "whiteEdge": 2 }
```

- **corners** are the 4 outer corners of the 64 squares. They are not the frame, not a printed border or coordinate
  ring, and not the table. The coordinates are pixels of the stored file, with pixel centres at integer coordinates.
- **Corner order** is clockwise on screen, starting from the corner with the smallest x + y. `promote` puts the
  corners into this order and remaps whiteEdge to match, so the order in the draft doesn't matter.
- **whiteEdge k** means the white pieces start along the edge corners[k] → corners[k+1]. Then a1 = corners[k+1]
  and h1 = corners[k], and a1 must be a dark square. The grid overlay labels A1/H1/H8/A8 and outlines a1, so check it
  there.
- Negative images (`neg-*.jpg`, no chessboard) get `"corners": null`.
- Target accuracy is under 1% of the board diagonal for every corner (about 8 px on a 1600 px photo). For corners
  hidden behind pieces, extrapolate along the visible boundary lines. The crops extend the projected boundary lines
  by one cell to help with this.

`corners.draft.json` is committed next to `corners.json` and holds the working state. Each entry has the same fields
plus the following:

| Field | Meaning |
|---|---|
| `source` | `detector`, `none` (the detector failed, annotate by hand) or `manual` (set this when you move the corners) |
| `confidence`, `detectSize`, `sizeSpread` | Detector confidence, the long side it ran at, and how much detections at 640 px and 960 px disagree (% of the diagonal) |
| `refine` | Lattice refinement: inlier saddle points, RMS residual in px, and the shift from the raw detection (%) |
| `whiteEdgeMethod`, `whiteEdgeTracker`, `whiteEdgeHeuristic`, `heuristicMargin` | How whiteEdge was guessed. `tracker` is the app's `OccupancyTracker` orientation from the starting position. `heuristic` is band brightness, used when the tracker cannot orient, and must be checked by eye. |
| `flags` | Things you must look at |
| `reviewed`, `reviewer` | Set by the annotator once all 4 corners and whiteEdge are confirmed on the crops |
| `crossChecked`, `crossChecker` | Set by the second agent |
| `notes` | Free text, for example "C2 hidden by the rook, extrapolated along both boundary lines" |

## Steps

Run every command from the repo root. `ANNOT_DIR` defaults to `tests/fixtures/web`.

### 1. Ingest

```sh
ANNOT=ingest ANNOT_RENAME=1 npm run annot
```

- Any image whose long side is over 1600 px is downscaled to 1600 px (JPEG q88, aspect ratio kept). The EXIF
  orientation is baked into the pixels, PNG files become JPEG, and metadata (EXIF/GPS/XMP) is stripped. Originals are
  copied to `tests/tools/annot-out/originals/web/` first. HEIC files are reported and skipped, so convert them to
  JPEG first.
- `ANNOT_RENAME=1` renames files to `NN-short-name.jpg`, numbering after the highest existing `NN-`. Names that
  already match the pattern, and `neg-*` files, are kept. The original name goes into `notes`. Leave the flag out to
  keep the names.
- It creates or completes `SOURCES.json` with one entry per image:
  `{file, title, author, sourceUrl, license, view, material, lighting, split, notes}`. Fields that are already
  filled are never overwritten. You can run ingest again at any time, for example after new photos arrive.

Then fill `view` (`oblique` or `topdown`), `material` (for example `brown-wood`, `green-vinyl`, `printed-black-grey`),
`lighting` (`window`, `overhead-glare`, `dim`, ...), and `title`/`author`/`license`/`sourceUrl` by hand. For the
user's own photos, use author "repo owner" and licence "CC0 (own photo)" unless told otherwise.

### 2. Split (after view and material are filled)

```sh
ANNOT=split npm run annot            # ANNOT_SEED=1 by default
```

This assigns `tune` or `holdout` to entries whose `split` is still null. About 1/3 of the images go to holdout,
stratified over view × material and seeded, so the result is deterministic. Existing values are never changed. The
bench skips holdout images unless `REALOCC_HOLDOUT=1`.

### 3. Propose corners and whiteEdge

```sh
ANNOT=propose npm run annot          # ANNOT_FILTER=<substring> for one image, ANNOT_FORCE=1 to redo unreviewed entries
```

For each image, propose does the following:

1. Runs the app's detector (`Detector.detect`, the bench's code path) at 640 px and 960 px and keeps the most
   confident detection.
2. Refines the grid on the full-resolution image: sub-pixel saddle points of the inner 7×7 lattice, then a
   least-squares homography with outliers rejected.
3. Guesses whiteEdge with the app's `OccupancyTracker`.
4. Writes `corners.draft.json` and renders the crops (step 4).

Reviewed entries are never touched. Entries where detection failed get `corners: null`, `source: "none"` and a flag.

### 4. Review each corner on zoomed crops

For every image, open these files from `tests/tools/annot-out/web/`:

- `<stem>.preview.png`: the whole image (1000 px) with the 8×8 grid, A1/H1/H8/A8 labels with corner indices
  `C0..C3`, the white edge in cyan, and a1 outlined. Check first that the grid covers exactly the 64 squares, not one
  cell too far into a border ring. Then check that A1 sits at white's left and on a dark square.
- `<stem>.c<k>.png` (k = 0..3): a 200 × 200 px crop around corner k, zoomed ×3. It shows a red crosshair on the
  proposal, the projected grid lines in yellow (boundary lines extended one cell past the corner), the white edge in
  cyan, and faint ticks every 10 image px with labels every 50 px (absolute coordinates). A green cross marks the
  `corners.json` corner, if there is one.
- `<stem>.c<k>.wide.png`: a 600 × 600 px context crop (×1) around the same corner.

To adjust a corner, read the true position off the ticks. At ×3 zoom, 1 image px is 3 display px, and the 10 px ticks
are 30 display px apart. Edit the corner in `corners.draft.json` and set `"source": "manual"`, then re-render and
check again:

```sh
ANNOT=crops ANNOT_FILTER=07-red npm run annot                   # re-render from the edited draft
ANNOT=crops ANNOT_FILTER=07-red ANNOT_CROP=80 ANNOT_ZOOM=6 npm run annot   # closer look
```

When the detector failed (`corners: null`), estimate the 4 corners on the preview or the original image and crop
there:

```sh
ANNOT=at ANNOT_FILTER=09-glass ANNOT_AT="412,300;1190,288;1300,860;350,880" npm run annot
```

This writes `<stem>.at-<x>-<y>.png` crops and an `<stem>.at-preview.png` quad preview. Refine the points, then write
them into the draft entry (any order, with `source: "manual"` and your whiteEdge) and run `ANNOT=crops` as above.

whiteEdge is a visual judgement. The white pieces stand along edge corners[k] → corners[k+1]. If
`whiteEdgeMethod` is `heuristic`, or a flag says the tracker and the heuristic disagree, decide from the preview
yourself. Once all 4 corners and whiteEdge are confirmed, set `"reviewed": true, "reviewer": "<agent name>"`. For
`neg-*` images, confirm there is no board and set `reviewed` with `corners` left null.

### 5. Cross-check (second agent)

The second agent reviews the other agent's reviewed entries on the same crops (`ANNOT=crops`) and the preview. If
the second agent annotates independently in a copy, compare the two files:

```sh
ANNOT=compare ANNOT_A=/path/to/mine.json ANNOT_B=tests/fixtures/web/corners.draft.json npm run annot
```

This prints the mean/max corner error (% of the diagonal) and the per-corner px error for each image, and whether
the two whiteEdges put a1/h1/h8/a8 on the same corners, so corner relabellings are accounted for. Resolve every
corner above 1% and every whiteEdge mismatch, then set `"crossChecked": true, "crossChecker": "<agent>"`.

### 6. Promote and render the grid overlays

```sh
ANNOT=promote npm run annot          # ANNOT_FORCE=1 to overwrite a differing corners.json entry
REALOCC_GRID_ONLY=1 REALOCC_DIRS=tests/fixtures/web BENCH_ONLY=realocc npm run bench
```

`promote` copies the entries with `"reviewed": true` into `corners.json` as `{width, height, corners, whiteEdge}`.
It normalises the corner order and warns about entries that were not cross-checked. It never overwrites a different
existing entry unless `ANNOT_FORCE=1`. The grid overlays, including holdout images, are written to
`tests/synth/realocc-out/png/web__<stem>.grid.png`. Look at every one, since these are also what the user reviews.

Then run the baseline:

```sh
BENCH_ONLY=realocc REALOCC_DIRS=tests/fixtures/web npm run bench                 # tune images
REALOCC_HOLDOUT=1 BENCH_ONLY=realocc npm run bench                               # everything, holdout included
ANNOT=compare ANNOT_B=tests/fixtures/web/corners.json npm run annot              # draft vs promoted (sanity)
```

### Commit

Stage only your own paths: the images, `SOURCES.json`, `corners.draft.json` and `corners.json`. Do not stage
`tests/tools/annot-out/`, which is gitignored. Run `git pull --rebase origin main` before `git push origin main`.

## Validation of the tools on tests/fixtures/real

The 5 photos and the negative were copied to a scratch directory, ingested (1920 → 1600 px) and proposed. The
proposals were then compared against the hand annotation in `tests/fixtures/real/corners.json`, which is itself
accurate to about 3 px (±5 px for hidden corners):

| Image | Mean err % diag | Max err % diag | whiteEdge (tracker) |
|---|---|---|---|
| 01-front-oblique-high | 0.45 | 0.69 | correct |
| 02-diagonal-oblique | 0.26 | 0.48 | correct |
| 03-near-overhead-rotated | 0.49 | 0.88 | correct |
| 04-side-low-oblique | 0.52 | 0.77 | correct |
| 05-dim-light-diagonal | 0.33 | 0.43 | correct (see below) |
| neg-01-tablecloth-tiles | no detection (correct) | | |

The band heuristic also gave the right whiteEdge on all 5 photos. For 05, the draft starts at a different corner:
two corners have almost the same x + y, and the draft follows the smallest-x+y rule strictly. Its whiteEdge (1) is
therefore the same edge as the GT's 2. `compare` reports it as a match.

**Proposals are not ground truth.** Errors of 0.3–0.9% are close to the 1% budget. A board with a printed border
ring, or one the detector locks one cell off, will be further out. Every corner must be confirmed on the crops.
