#!/usr/bin/env python3
"""Batch Clinical Screening & Diagnostic Report Generator.
Processes an input directory of color fundus images, executes multi-model inference,
computes Grad-CAM explainability, and generates a structured summary CSV
along with individual 1-page clinical diagnostic PDF reports using ReportLab.
"""

import argparse
import base64
import csv
import datetime
import io
import json
import os
import sys
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from PIL import Image

# Ensure project root in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set expandable segments
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from src.web.inference_engine import InferenceEngine, MODEL_REGISTRY

# ReportLab imports
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image as RLImage,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


def generate_patient_pdf(result: dict, original_img_path: Path, output_pdf_path: Path):
    """Generate publication/clinical-grade 1-page diagnostic PDF report."""
    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ClinicTitle",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#0f172a")
    )
    section_heading = ParagraphStyle(
        "SectionHeading",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=13,
        textColor=colors.HexColor("#0369a1")
    )
    body_style = ParagraphStyle(
        "BodyTextCustom",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#1e293b")
    )
    bold_label = ParagraphStyle(
        "BoldLabel",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0f172a")
    )

    story = []

    # 1. Header Banner
    header_data = [
        [
            Paragraph("<b>JADAVPUR UNIVERSITY</b><br/><font size='9'>DEPARTMENT OF INFORMATION TECHNOLOGY</font>", title_style),
            Paragraph("<b>RETINAL SCREENING REPORT</b><br/><font size='8' color='#64748b'>Automated Multi-Architecture Diagnostic System</font>", ParagraphStyle("RightHeader", parent=title_style, alignment=2))
        ]
    ]
    t_header = Table(header_data, colWidths=[3.8 * inch, 3.8 * inch])
    t_header.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
    ]))
    story.append(t_header)
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#0284c7"), spaceAfter=8, spaceBefore=4))

    # 2. Patient & Acquisition Metadata Card
    patient_id = f"PAT-{original_img_path.stem.upper()[:12]}"
    date_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    top_pred = result["top_prediction"]
    uncert = result["uncertainty"]

    meta_table_data = [
        [
            Paragraph("<b>Sample / Image ID:</b>", bold_label),
            Paragraph(original_img_path.name, body_style),
            Paragraph("<b>Diagnostic Date:</b>", bold_label),
            Paragraph(date_str, body_style)
        ],
        [
            Paragraph("<b>Patient Identifier:</b>", bold_label),
            Paragraph(patient_id, body_style),
            Paragraph("<b>Imaging Modality:</b>", bold_label),
            Paragraph("Color Fundus Photography (CFP)", body_style)
        ],
        [
            Paragraph("<b>Screening Model:</b>", bold_label),
            Paragraph(f"{result['model_name']} ({result['model_accuracy']})", body_style),
            Paragraph("<b>Uncertainty Rating:</b>", bold_label),
            Paragraph(f"{uncert['certainty_level']} (Entropy: {uncert['normalized_entropy']})", body_style)
        ]
    ]
    t_meta = Table(meta_table_data, colWidths=[1.5 * inch, 2.3 * inch, 1.5 * inch, 2.3 * inch])
    t_meta.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 0.8, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 8))

    # 3. Diagnostic Verdict Hero Card
    urgency_color = "#dc2626" if top_pred["urgency_level"] in ["red", "crimson"] else ("#d97706" if top_pred["urgency_level"] == "amber" else "#059669")
    verdict_data = [
        [
            Paragraph(f"<font color='{urgency_color}' size='11'><b>PRIMARY DIAGNOSIS: {top_pred['short_name'].upper()}</b></font><br/><font size='8' color='#475569'>Confidence: <b>{top_pred['percentage']}%</b> &bull; Triage Priority: <font color='{urgency_color}'><b>{top_pred['urgency'].upper()}</b></font></font>", body_style)
        ]
    ]
    t_verdict = Table(verdict_data, colWidths=[7.6 * inch])
    t_verdict.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f1f5f9")),
        ("BOX", (0, 0), (-1, -1), 1.2, colors.HexColor(urgency_color)),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
    ]))
    story.append(t_verdict)
    story.append(Spacer(1, 8))

    # 4. Side-by-Side Images: Input CLAHE Fundus vs Grad-CAM Overlay
    cam_data = result.get("gradcam")
    if cam_data and "preprocessed_b64" in cam_data and "overlay_b64" in cam_data:
        pre_bytes = base64.b64decode(cam_data["preprocessed_b64"].split(",")[1])
        over_bytes = base64.b64decode(cam_data["overlay_b64"].split(",")[1])

        rl_pre = RLImage(io.BytesIO(pre_bytes), width=2.6 * inch, height=2.6 * inch)
        rl_over = RLImage(io.BytesIO(over_bytes), width=2.6 * inch, height=2.6 * inch)

        img_table_data = [
            [rl_pre, rl_over],
            [
                Paragraph("<b>Figure A:</b> Enhanced Input (CLAHE Contrast)", ParagraphStyle("Cap1", parent=body_style, alignment=1)),
                Paragraph("<b>Figure B:</b> Grad-CAM Anatomical Focus Overlay", ParagraphStyle("Cap2", parent=body_style, alignment=1))
            ]
        ]
        t_imgs = Table(img_table_data, colWidths=[3.8 * inch, 3.8 * inch])
        t_imgs.setStyle(TableStyle([
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(t_imgs)
        story.append(Spacer(1, 8))

    # 5. Ranked Differential Diagnosis Table (Top 4)
    story.append(Paragraph("Differential Diagnostic Probabilities (Ranked Spectrum)", section_heading))
    story.append(Spacer(1, 3))

    diff_table_data = [["Rank", "Pathology Condition", "Diagnostic Probability", "Clinical Triage Risk"]]
    for item in result["distribution"][:4]:
        diff_table_data.append([
            str(item["rank"]),
            item["short_name"],
            f"{item['percentage']}%",
            item["urgency"]
        ])

    t_diff = Table(diff_table_data, colWidths=[0.6 * inch, 3.2 * inch, 1.8 * inch, 2.0 * inch])
    t_diff.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f172a")),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 7.5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5),
        ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
    ]))
    story.append(t_diff)
    story.append(Spacer(1, 8))

    # 6. Clinical Management & Recommended Actions
    story.append(Paragraph("Clinical Observations & Recommended Actions", section_heading))
    story.append(Spacer(1, 3))

    desc_p = Paragraph(f"<b>Pathological Profile:</b> {top_pred['description']}", body_style)
    story.append(desc_p)
    story.append(Spacer(1, 2))

    recs = top_pred.get("recommendations", [])
    rec_text = "<b>Recommended Ophthalmic Follow-up:</b> " + " | ".join(recs[:3])
    story.append(Paragraph(rec_text, body_style))
    story.append(Spacer(1, 10))

    # 7. Signature Block
    sig_data = [
        [
            Paragraph("<i>Automated Artificial Intelligence Screening Aid.<br/>Must be correlated with comprehensive slit-lamp/OCT examination.</i>", ParagraphStyle("Disclaimer", parent=body_style, fontSize=6.5, textColor=colors.HexColor("#64748b"))),
            Paragraph("<b>Reviewed by Examining Ophthalmologist:</b><br/><br/>____________________________________<br/>Date: _______________", ParagraphStyle("Sig", parent=body_style, alignment=2, fontSize=7.5))
        ]
    ]
    t_sig = Table(sig_data, colWidths=[4.2 * inch, 3.4 * inch])
    t_sig.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "BOTTOM"),
    ]))
    story.append(t_sig)

    doc.build(story)


def main():
    parser = argparse.ArgumentParser(description="Batch Clinical Screening and Report Generator")
    parser.add_argument("--input-dir", type=str, required=True, help="Path to folder of fundus images to screen")
    parser.add_argument("--output-dir", type=str, default="outputs/batch_screening_results", help="Directory to save output reports")
    parser.add_argument("--benchmark", type=str, default="10class", choices=["10class", "4class"], help="Benchmark dataset profile")
    parser.add_argument("--model", type=str, default="ensemble_quad", help="Model ID (e.g. ensemble_quad, efficientnet_b3, convnext_small)")
    parser.add_argument("--max-images", type=int, default=50, help="Maximum number of images to process (default 50)")
    parser.add_argument("--generate-pdf", action="store_true", default=True, help="Generate 1-page clinical PDF report per image")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_dir = output_dir / "patient_reports"
    pdf_dir.mkdir(parents=True, exist_ok=True)

    if not input_dir.exists():
        print(f"Error: Input directory {input_dir} does not exist.")
        sys.exit(1)

    exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG"]
    image_paths = []
    for ext in exts:
        image_paths.extend(list(input_dir.glob(ext)))
        image_paths.extend(list(input_dir.glob(f"**/{ext}")))

    image_paths = sorted(list(set(image_paths)))
    if not image_paths:
        print(f"No image files found in {input_dir}")
        sys.exit(1)

    image_paths = image_paths[:args.max_images]
    print("=" * 75)
    print(" AUTOMATED CLINICAL BATCH SCREENING ENGINE")
    print(f" Input Directory: {input_dir}")
    print(f" Output Directory: {output_dir}")
    print(f" Benchmark: {args.benchmark} | Model: {args.model}")
    print(f" Total Images Queued: {len(image_paths)}")
    print("=" * 75)

    engine = InferenceEngine()
    summary_records = []
    csv_path = output_dir / "screening_summary.csv"

    for idx, img_path in enumerate(image_paths):
        print(f"[{idx+1}/{len(image_paths)}] Processing: {img_path.name} ...", end="", flush=True)

        try:
            bgr = cv2.imread(str(img_path))
            if bgr is None:
                print(" FAILED (unreadable)")
                continue

            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = engine.predict(
                rgb_image=rgb,
                benchmark=args.benchmark,
                model_id=args.model,
                generate_cam=args.generate_pdf
            )

            top = result["top_prediction"]
            uncert = result["uncertainty"]

            summary_records.append({
                "Image_Name": img_path.name,
                "Relative_Path": str(img_path.relative_to(input_dir)),
                "Predicted_Diagnosis": top["short_name"],
                "Confidence_Pct": top["percentage"],
                "Urgency": top["urgency"],
                "Urgency_Level": top["urgency_level"],
                "Normalized_Entropy": uncert["normalized_entropy"],
                "Certainty_Level": uncert["certainty_level"],
                "Model_Evaluated": result["model_name"]
            })

            if args.generate_pdf:
                pdf_file = pdf_dir / f"Report_{img_path.stem}.pdf"
                generate_patient_pdf(result, img_path, pdf_file)

            print(f" -> {top['short_name']} ({top['percentage']}%) [{uncert['certainty_level']}]")

        except Exception as e:
            print(f" ERROR: {e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "Image_Name", "Relative_Path", "Predicted_Diagnosis", "Confidence_Pct",
            "Urgency", "Urgency_Level", "Normalized_Entropy", "Certainty_Level", "Model_Evaluated"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_records)

    print("\n" + "=" * 75)
    print(" BATCH SCREENING COMPLETE")
    print(f" Successfully Screened: {len(summary_records)} / {len(image_paths)} fundus images")
    print(f" Summary CSV: {csv_path.resolve()}")
    if args.generate_pdf:
        print(f" Individual Patient Reports: {pdf_dir.resolve()}")
    print("=" * 75)


if __name__ == "__main__":
    main()
