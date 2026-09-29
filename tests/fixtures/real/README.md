# Real board photos

These are phone photos taken by the repo owner with a TECNO CM5, originally at 3840×2160. They were downscaled to 1920×1080 (06: 1080×1920, portrait) and their metadata was stripped.

All six show the same board: a black/light-grey printed playing surface inside a thin black ring printed with coordinates (A–H, 1–8), set in a reddish wooden frame on a grey wood-effect table. Pieces are in the starting position.

| File | View | Notes |
|---|---|---|
| 01-front-oblique-high.jpg | Seated, fairly high, board roughly frontal | Warm artificial light, notebook and plant in frame |
| 02-diagonal-oblique.jpg | Oblique, board rotated about 45° | The board's corners point up, down, left and right |
| 03-near-overhead-rotated.jpg | Close to overhead, rotated about 30° | Weak perspective |
| 04-side-low-oblique.jpg | Low oblique from the side | Strong perspective; tall pieces hide rows 1–2 and 7–8; board side and thickness visible |
| 05-dim-light-diagonal.jpg | Oblique, rotated | Dim light, visible sensor noise, slight blur |
| 06-overhead-portrait-floor.jpg | Near overhead, **portrait** 1080×1920 (from 2160×3840), board in the upper half | Board on grey floor tiles; a white tube and a black box in the lower half. White pieces on the far (top) edge, on the ranks printed 7–8. Dark squares have lighter mottled patches; the black pieces are glossy with bright specular rims |
| neg-01-tablecloth-tiles.jpg | **Negative (no board)** | A hard negative. A tablecloth with a strong repeating geometric line grid (squares, octagons, diagonals, gold lines), and large floor tiles forming a perspective grid. Warm, dim light with glare. The detector must return `null`. |

Files starting with `neg-` contain no chessboard. They have no `corners.json` entry, and any detection on them counts as a false positive.

## Why these are harder than the synthetic set
- **The printed black ring next to the playing area looks like more dark squares.** It is the same colour as the dark squares and sits right on the outer grid lines, so a comb fit can slip one cell outward. The checker-verification bonus for "the margin is not checkered" is what has to reject that.
- **In the starting position the outer two ranks are covered by pieces.** The outermost grid lines (ranks 1/8) are mostly hidden. The middle 4×8 area is clean, so the grid has to be extrapolated outward and then checked.
- **The table has a wood-grain pattern** that produces many long, weak parallel edges.
- **Low light (05)** means noise and a flat, low-contrast image.

## Ground truth
`corners.json` holds the 4 corners of the 8x8 playing area in each photo's pixel coordinates, in clockwise order starting top-left. These are the outer corners of the 64 squares, not the black ring or the wooden frame. The file is to be produced by carefully annotating zoomed crops.
