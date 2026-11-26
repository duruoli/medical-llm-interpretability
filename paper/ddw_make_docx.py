"""Build a .docx of the DDW abstract (title, body, two figures with captions) for advisor review.
Run: python paper/ddw_make_docx.py  -> paper/DDW2027_abstract_draft.docx"""
import re
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Inches, Pt, RGBColor

HERE = Path(__file__).parent
md = (HERE / "ddw2027_abstract.md").read_text()
title = re.search(r"## Title\n\n(.*)", md).group(1).strip()
body = md.split("<!-- BODY START -->")[1].split("<!-- BODY END -->")[0].strip()

CAP1 = ("Figure 1. Shift in the model's belief caused by each test result, divided by the shift predicted "
        "by the literature likelihood ratio (dashed line = 1, exactly as predicted). Orange: positive report; "
        "blue: negative report. For CT, negative reports reach only 33-68% of the predicted shift while positive "
        "reports meet or exceed it; for ultrasound, negative reports are near target but positive reports "
        "overshoot 2-3-fold. Negative lab results are not under-weighted. Qwen3.5-9B, 94 held-out patients.")
CAP2 = ("Figure 2. Positive/negative imbalance on held-out patients before (gray) and after (green) the weight "
        "edit; 1x means the ratio of the positive shift to the negative shift matches the ratio predicted by "
        "the literature. *Tests never used to develop the edit. Random edits of the same size left the "
        "overall imbalance at 2.4-2.5x; negative results that wrongly raised belief fell from 79 to 2.")

doc = Document()
st = doc.styles["Normal"]
st.font.name, st.font.size = "Arial", Pt(11)


def note(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.italic, r.font.size, r.font.color.rgb = True, Pt(9), RGBColor(0x52, 0x51, 0x4E)


doc.add_heading("DDW 2027 abstract - draft for review", level=1)
note(f"Deadline: Wed Dec 2, 2026, 9:00 pm ET. Limits: title <= 255 characters, ALL CAPS; body <= 2,900 "
     f"characters including spaces and headings; up to 2 images. Current: title {len(title)}, body {len(body)}.")

doc.add_heading("Title", level=2)
doc.add_paragraph(title).runs[0].bold = True

doc.add_heading("Authors", level=2)
doc.add_paragraph("Duruo Li, Dennis L. Shung, Bradly C. Stadie")
note("Entered separately in the submission system (not part of the title or body). Affiliations to be added.")

doc.add_heading("Abstract", level=2)
for para in body.split("\n\n"):
    p = doc.add_paragraph()
    head, _, rest = para.partition(":")
    p.add_run(head + ":").bold = True
    p.add_run(rest)

doc.add_heading("Figures", level=2)
for img, cap in [("ddw_fig1_shifts.png", CAP1), ("ddw_fig2_repair.png", CAP2)]:
    doc.add_picture(str(HERE / img), width=Inches(6.3))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = doc.add_paragraph(cap)
    c.runs[0].font.size = Pt(9.5)

out = HERE / "DDW2027_abstract_draft.docx"
doc.save(out)
print(out, "title", len(title), "body", len(body))
