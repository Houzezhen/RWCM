"""Convert the Inter OTF (CFF outlines) files to TrueType so matplotlib embeds them as clean Type 42 fonts."""
from pathlib import Path
import glob

from fontTools.ttLib import TTFont, newTable
from fontTools.pens.cu2quPen import Cu2QuPen
from fontTools.pens.ttGlyphPen import TTGlyphPen

OUT = Path(__file__).resolve().parents[1] / "assets" / "fonts"
OUT.mkdir(parents=True, exist_ok=True)
SRC = "A:/texlive/2026/texmf-dist/fonts/opentype/public/inter/Inter-{}.otf"

for style in ("Regular", "Medium", "SemiBold", "Bold"):
    font = TTFont(SRC.format(style))
    order = font.getGlyphOrder()
    gs = font.getGlyphSet()
    glyf = newTable("glyf"); glyf.glyphOrder = order; glyf.glyphs = {}
    for name in order:
        pen = TTGlyphPen(gs)
        gs[name].draw(Cu2QuPen(pen, 1.0, reverse_direction=True))
        glyf.glyphs[name] = pen.glyph()
    font["glyf"] = glyf
    font["loca"] = newTable("loca")
    font["maxp"].tableVersion = 0x00010000
    for attr in ("maxZones", "maxTwilightPoints", "maxStorage", "maxFunctionDefs", "maxInstructionDefs",
                 "maxStackElements", "maxSizeOfInstructions", "maxComponentElements"):
        setattr(font["maxp"], attr, 0)
    font["maxp"].maxZones = 1
    font["maxp"].maxComponentDepth = 0
    font["post"].formatType = 2.0
    font["post"].extraNames = []
    font["post"].mapping = {}
    font["head"].glyphDataFormat = 0
    del font["CFF "]
    if "VORG" in font:
        del font["VORG"]
    font.sfntVersion = "\x00\x01\x00\x00"
    font.save(OUT / f"Inter-{style}.ttf")
    print("converted", style)
