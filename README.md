# quilting-svg2pdf
Takes a paper pieced design (in specific format), separates all the segments and produces a PDF ready for printing with seam allowances marked on.

## Installation

Requires Python3.6+.

`pip install -r requirements.txt`

## Usage

`python3 extract.py svg_file_name.svg`

## CURRENT STATUS
At the moment, this program will take your SVG in the format detailed below, and create a separate SVG image file for each segment, with a 1/4 inch seam allowance marked on. I am working on putting the pattern pieces onto A4/Letter paper and returning a ready-to-print PDF.

## SVG File Format
Your paper piece pattern should use black (`#000000`) lines as the separators, and only use paths (i.e do not use the rectangle, circle or polygon tools, use the line tool only). Then, draw a red line (`#ff0000`, I add transparency to see the black lines underneath) over each of the black lines that mark segment boundaries. Finally, make small alignment markings between segments using blue lines (`#0000ff`). These colours are imporant, and how the program differentiates between normal lines, segment boundary lines, and markers. Fill colours and text can be used and will be transferred onto the final separated segment pieces correctly. Make sure the SVG is only one layer.

An example file will be uploaded soon.

### TODO List
These are the current TODO's:
- Add support for `translate`, `rotate` and `skew` on paths (currently only `scale` and `matrix` are supported).
- Work out whether multiple of the same transform are allowed on one path, in the SVG standard (I sincerely hope not) and support if so.
- Add support for all other transform types on the main image (currently only supports `translate`).
- Add support for other shapes, not just `path` (e.g `rect` etc).
- Work out how to bin-pack pattern pieces onto given paper size.
- Once bin-packed, output as a PDF.
