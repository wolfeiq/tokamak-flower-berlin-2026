"""Export the six-slide talk with browser artwork and editable speaker notes.

First run: node presentation/harness/export.mjs --powerpoint
The worked example is a static diagram. No media playback is needed.
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

from pptx import Presentation

ROOT = Path(__file__).resolve().parent
EXPORTS = ROOT / ".exports"
TITLES = [
    "What it is and who it is for",
    "What goes in and what comes out",
    "Why use an agent here?",
    "Inside one Flower AgentApp run",
    "The heating example",
    "What we can claim and what comes next",
]


def build():
    for index in range(1, 7):
        if not (EXPORTS / f"slide-{index:02}.png").is_file():
            raise SystemExit("Run export.mjs --powerpoint before building PowerPoint.")

    speech = (ROOT / "speech_for_harness.txt").read_text(encoding="utf-8")
    sections = re.split(r"\n\d+\. [^\n]+\n", speech)[1:]
    assert len(sections) == 6
    deck = Presentation()
    # Exactly 16:9 in native EMU units.
    deck.slide_width, deck.slide_height = 12192000, 6858000
    deck.core_properties.title = "Fusion Investigator — Flower agent harness"
    deck.core_properties.subject = "Five-minute harness explanation and synthetic worked example"
    deck.core_properties.author = "Fusion Investigator project"
    deck.core_properties.keywords = "Flower, AgentApp, fusion, synthetic, investigation"

    for index, (title, section) in enumerate(zip(TITLES, sections), start=1):
        slide = deck.slides.add_slide(deck.slide_layouts[6])
        slide.name = title
        picture = slide.shapes.add_picture(
            str(EXPORTS / f"slide-{index:02}.png"), 0, 0,
            deck.slide_width, deck.slide_height,
        )
        picture.name = f"Slide {index} — {title}"
        picture._element.xpath(".//p:cNvPr")[0].set(
            "descr", re.sub(r"\[[^]]*\]", "", section).strip()
        )
        slide.notes_slide.notes_text_frame.text = section.strip()

    output = ROOT / "fusion-investigator.pptx"
    # Build and validate separately so a failed build cannot corrupt the last deck.
    temporary = EXPORTS / "fusion-investigator.pptx"
    deck.save(temporary)
    with zipfile.ZipFile(temporary) as package:
        assert package.testzip() is None
        names = package.namelist()
        slides = [n for n in names if re.fullmatch(r"ppt/slides/slide\d+\.xml", n)]
        notes = [n for n in names if re.fullmatch(r"ppt/notesSlides/notesSlide\d+\.xml", n)]
        assert len(slides) == len(notes) == 6
        assert not any(n.endswith(".mp4") for n in names)
    reopened = Presentation(temporary)
    for slide, section in zip(reopened.slides, sections):
        assert slide.notes_slide.notes_text_frame.text == section.strip()
    temporary.replace(output)
    print(json.dumps({"output": str(output), "slides": 6, "speaker_notes": 6,
                      "embedded_video": False, "size_bytes": output.stat().st_size}))


if __name__ == "__main__":
    build()
