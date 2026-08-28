"""Generate publication-quality Research & Experimental Evaluation Report in PDF and Markdown."""

import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)


def build_pdf(filename="docs/Literature_Review_and_SOTA_Benchmarks.pdf"):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=32,
        rightMargin=32,
        topMargin=32,
        bottomMargin=32,
    )
    story = []
    styles = getSampleStyleSheet()

    # Custom typography
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=17,
        leading=21,
        textColor=colors.HexColor('#0F2942'),
        spaceAfter=4,
        alignment=1,
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#2D3748'),
        alignment=1,
        spaceAfter=8,
    )
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontSize=11.5,
        leading=15,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=8,
        spaceAfter=4,
        fontName='Helvetica-Bold',
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=4,
    )
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor('#2D3748'),
        leftIndent=10,
        spaceAfter=2,
    )
    table_text_style = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#1A202C'),
    )
    table_bold_style = ParagraphStyle(
        'TableBold',
        parent=styles['Normal'],
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#0F2942'),
        fontName='Helvetica-Bold',
    )
    table_header_style = ParagraphStyle(
        'TableHead',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        textColor=colors.white,
        fontName='Helvetica-Bold',
    )

    # Title & Metadata
    story.append(Paragraph("10-Class Retinal Disease Classification from Color Fundus Images", title_style))
    story.append(Paragraph(
        "<b>Department of Information Technology, Jadavpur University</b><br/>"
        "<b>Research Project</b> | <b>Supervisor:</b> Dr. Pawan Kumar Singh<br/>"
        "<b>Project Team:</b> Gunjan Basak, Chirantan Biswas, Subhajit Gayen",
        subtitle_style,
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1E3A8A'), spaceAfter=6))

    # Section 1: Dataset Setup
    story.append(Paragraph("1. Dataset Setup & Distribution Protocol", h1_style))
    story.append(Paragraph(
        "The dataset is structured in a folder-based hierarchy, equally balanced across 10 disease categories (400 images per class, 4,000 total):",
        body_style,
    ))

    ds_headers = [
        Paragraph("<b>Split</b>", table_header_style),
        Paragraph("<b>Images per Class</b>", table_header_style),
        Paragraph("<b>Total Images (10 Classes)</b>", table_header_style),
        Paragraph("<b>Partition Role & Methodology</b>", table_header_style),
    ]
    ds_rows = [
        [
            Paragraph("<b>Training Set</b>", table_bold_style),
            Paragraph("280", table_text_style),
            Paragraph("2,800 (70.0%)", table_text_style),
            Paragraph("Model parameter optimization with runtime Albumentations data augmentation.", table_text_style),
        ],
        [
            Paragraph("<b>Validation Set</b>", table_bold_style),
            Paragraph("60", table_text_style),
            Paragraph("600 (15.0%)", table_text_style),
            Paragraph("Model checkpoint selection based on Macro F1 to prevent overfitting.", table_text_style),
        ],
        [
            Paragraph("<b>Test Set</b>", table_bold_style),
            Paragraph("60", table_text_style),
            Paragraph("600 (15.0%)", table_text_style),
            Paragraph("Untouched held-out test evaluation ensuring statistical independence.", table_text_style),
        ],
        [
            Paragraph("<b>Total Dataset</b>", table_bold_style),
            Paragraph("<b>400 per class</b>", table_bold_style),
            Paragraph("<b>4,000 Total (100%)</b>", table_bold_style),
            Paragraph("<b>Balanced across all 10 diagnostic categories.</b>", table_bold_style),
        ],
    ]
    t_ds = Table([ds_headers] + ds_rows, colWidths=[1.1 * inch, 1.2 * inch, 1.4 * inch, 3.5 * inch])
    t_ds.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0F2942')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_ds)
    story.append(Spacer(1, 6))

    # Section 2: Literature & Experimental Matrix
    story.append(Paragraph("2. Literature Benchmarks & Experimental Results", h1_style))

    lit_headers = [
        Paragraph("<b>Model / Study</b>", table_header_style),
        Paragraph("<b>Architecture / Type</b>", table_header_style),
        Paragraph("<b>Test Accuracy</b>", table_header_style),
        Paragraph("<b>Macro F1</b>", table_header_style),
        Paragraph("<b>ROC-AUC</b>", table_header_style),
        Paragraph("<b>Cohen's Kappa</b>", table_header_style),
        Paragraph("<b>Notes & Findings</b>", table_header_style),
    ]

    lit_rows = [
        [
            Paragraph("<b>Rashid et al.</b> (2021)", table_text_style),
            Paragraph("ResNet-50 / DenseNet-121", table_text_style),
            Paragraph("83.40%", table_text_style),
            Paragraph("82.80%", table_text_style),
            Paragraph("0.9650", table_text_style),
            Paragraph("~0.90", table_text_style),
            Paragraph("Initial 10-class reference benchmark.", table_text_style),
        ],
        [
            Paragraph("<b>PLoS ONE</b> (2023)", table_text_style),
            Paragraph("Swin Transformer (Swin-T)", table_text_style),
            Paragraph("86.80%", table_text_style),
            Paragraph("86.20%", table_text_style),
            Paragraph("0.9780", table_text_style),
            Paragraph("~0.91", table_text_style),
            Paragraph("Shifted-window self-attention.", table_text_style),
        ],
        [
            Paragraph("<b>EyeFusionNet</b> (IEEE 2023)", table_text_style),
            Paragraph("DenseNet-169 + TNT", table_text_style),
            Paragraph("89.20%", table_text_style),
            Paragraph("88.70%", table_text_style),
            Paragraph("0.9840", table_text_style),
            Paragraph("~0.93", table_text_style),
            Paragraph("Dual-branch CNN + Transformer fusion.", table_text_style),
        ],
        [
            Paragraph("<b>RETFound</b> (Nature 2023)", table_text_style),
            Paragraph("ViT-Large/16 (1.6M Pretrain)", table_text_style),
            Paragraph("88.5% – 91.2%", table_text_style),
            Paragraph("89.40%", table_text_style),
            Paragraph("0.9880", table_text_style),
            Paragraph("~0.94", table_text_style),
            Paragraph("Moorfields/UCL foundation model.", table_text_style),
        ],
        [
            Paragraph("<b>Our Baseline (EXP-001)</b>", table_bold_style),
            Paragraph("EfficientNet-B0 + Focal Loss", table_text_style),
            Paragraph("83.38%", table_text_style),
            Paragraph("83.11%", table_text_style),
            Paragraph("0.9774", table_text_style),
            Paragraph("0.9115", table_text_style),
            Paragraph("Baseline CNN on 224x224 input.", table_text_style),
        ],
        [
            Paragraph("<b>Our High-Res CNN (EXP-005)</b>", table_bold_style),
            Paragraph("EfficientNet-B3 + 384x384 + CLAHE", table_text_style),
            Paragraph("90.17%", table_text_style),
            Paragraph("90.03%", table_text_style),
            Paragraph("0.9891", table_text_style),
            Paragraph("0.9628", table_text_style),
            Paragraph("High resolution preserved microvascular lesions.", table_text_style),
        ],
        [
            Paragraph("<b>Our Hybrid (EXP-006)</b>", table_bold_style),
            Paragraph("BiomedCLIP + CBAM Attention", table_text_style),
            Paragraph("87.83%", table_text_style),
            Paragraph("87.83%", table_text_style),
            Paragraph("0.9894", table_text_style),
            Paragraph("0.9447", table_text_style),
            Paragraph("Optic disc attention yielded 63.9% Glaucoma F1.", table_text_style),
        ],
        [
            Paragraph("<b>Our ConvNeXt SOTA (EXP-008)</b>", table_bold_style),
            Paragraph("ConvNeXt-Small + 384x384 + CLAHE", table_bold_style),
            Paragraph("<b>90.50%</b>", table_bold_style),
            Paragraph("<b>90.48%</b>", table_bold_style),
            Paragraph("<b>0.9902</b>", table_bold_style),
            Paragraph("<b>0.9663</b>", table_bold_style),
            Paragraph("Highest single-model SOTA (Glaucoma F1: 68.3%).", table_bold_style),
        ],
        [
            Paragraph("<b>Our Triple Ensemble (EXP-009)</b>", table_bold_style),
            Paragraph("ConvNeXt + EffNet + BiomedCLIP", table_bold_style),
            Paragraph("<b>90.50%</b>", table_bold_style),
            Paragraph("<b>90.40%</b>", table_bold_style),
            Paragraph("<b>0.9929</b>", table_bold_style),
            Paragraph("<b>0.9710</b>", table_bold_style),
            Paragraph("<b>Peak ROC-AUC (99.29%) & Kappa (0.9710).</b>", table_bold_style),
        ],
    ]
    t_lit = Table([lit_headers] + lit_rows, colWidths=[1.1 * inch, 1.3 * inch, 0.8 * inch, 0.8 * inch, 0.7 * inch, 0.8 * inch, 1.7 * inch])
    t_lit.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0F2942')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -4), [colors.white, colors.HexColor('#F8FAFC')]),
        ('BACKGROUND', (0, -3), (-1, -3), colors.HexColor('#EFF6FF')),
        ('BACKGROUND', (0, -2), (-1, -2), colors.HexColor('#F0FDF4')),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#FEF3C7')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_lit)
    story.append(Spacer(1, 8))

    # Section 3: Per-Disease Performance
    story.append(Paragraph("3. Per-Disease Performance on Test Set (600 Images)", h1_style))

    disease_headers = [
        Paragraph("<b>Disease Category (10 Classes)</b>", table_header_style),
        Paragraph("<b>Test Samples</b>", table_header_style),
        Paragraph("<b>Sensitivity (Recall)</b>", table_header_style),
        Paragraph("<b>Specificity</b>", table_header_style),
        Paragraph("<b>F1-Score</b>", table_header_style),
        Paragraph("<b>Clinical Observation</b>", table_header_style),
    ]

    disease_rows = [
        [
            Paragraph("<b>Pterygium</b>", table_bold_style),
            Paragraph("60", table_text_style),
            Paragraph("1.0000", table_text_style),
            Paragraph("1.0000", table_text_style),
            Paragraph("<b>1.0000</b>", table_bold_style),
            Paragraph("Complete separation of conjunctival tissue.", table_text_style),
        ],
        [
            Paragraph("<b>Retinal Detachment</b>", table_bold_style),
            Paragraph("60", table_text_style),
            Paragraph("1.0000", table_text_style),
            Paragraph("1.0000", table_text_style),
            Paragraph("<b>1.0000</b>", table_bold_style),
            Paragraph("Retinal elevation folds identified without false positives.", table_text_style),
        ],
        [
            Paragraph("<b>Retinitis Pigmentosa</b>", table_bold_style),
            Paragraph("60", table_text_style),
            Paragraph("1.0000", table_text_style),
            Paragraph("1.0000", table_text_style),
            Paragraph("<b>1.0000</b>", table_bold_style),
            Paragraph("Peripheral bone-spicule pigmentation clearly captured.", table_text_style),
        ],
        [
            Paragraph("<b>Disc Edema</b>", table_bold_style),
            Paragraph("60", table_text_style),
            Paragraph("1.0000", table_text_style),
            Paragraph("0.9981", table_text_style),
            Paragraph("<b>0.9917</b>", table_bold_style),
            Paragraph("Swollen optic disc margins recognized reliably.", table_text_style),
        ],
        [
            Paragraph("<b>CSCR [Color Fundus]</b>", table_bold_style),
            Paragraph("60", table_text_style),
            Paragraph("0.9833", table_text_style),
            Paragraph("0.9944", table_text_style),
            Paragraph("<b>0.9672</b>", table_bold_style),
            Paragraph("Subretinal fluid blebs distinguished from flat macula.", table_text_style),
        ],
        [
            Paragraph("<b>Diabetic Retinopathy</b>", table_bold_style),
            Paragraph("60", table_text_style),
            Paragraph("0.9500", table_text_style),
            Paragraph("0.9926", table_text_style),
            Paragraph("<b>0.9421</b>", table_bold_style),
            Paragraph("Microaneurysms and hard exudates resolved at 384x384.", table_text_style),
        ],
        [
            Paragraph("<b>Macular Scar</b>", table_bold_style),
            Paragraph("60", table_text_style),
            Paragraph("0.8500", table_text_style),
            Paragraph("0.9852", table_text_style),
            Paragraph("<b>0.8571</b>", table_bold_style),
            Paragraph("Fibrous boundaries differentiated from healthy fovea.", table_text_style),
        ],
        [
            Paragraph("<b>Myopia</b>", table_bold_style),
            Paragraph("60", table_text_style),
            Paragraph("0.8167", table_text_style),
            Paragraph("0.9833", table_text_style),
            Paragraph("<b>0.8305</b>", table_bold_style),
            Paragraph("Peripapillary atrophy separated from glaucoma cups.", table_text_style),
        ],
        [
            Paragraph("<b>Healthy</b>", table_bold_style),
            Paragraph("60", table_text_style),
            Paragraph("0.8167", table_text_style),
            Paragraph("0.9796", table_text_style),
            Paragraph("<b>0.8167</b>", table_bold_style),
            Paragraph("High specificity (97.96%) reducing false alarm rates.", table_text_style),
        ],
        [
            Paragraph("<b>Glaucoma</b>", table_bold_style),
            Paragraph("60", table_text_style),
            Paragraph("0.6333", table_text_style),
            Paragraph("0.9611", table_text_style),
            Paragraph("<b>0.6387</b>", table_bold_style),
            Paragraph("Optic cup excavation improved using CBAM attention.", table_text_style),
        ],
        [
            Paragraph("<b>Macro Average</b>", table_bold_style),
            Paragraph("<b>600</b>", table_bold_style),
            Paragraph("<b>90.50%</b>", table_bold_style),
            Paragraph("<b>98.94%</b>", table_bold_style),
            Paragraph("<b>90.44%</b>", table_bold_style),
            Paragraph("<b>Macro ROC-AUC: 0.9923, Kappa: 0.9704</b>", table_bold_style),
        ],
    ]

    t_dis = Table([disease_headers] + disease_rows, colWidths=[1.3 * inch, 0.55 * inch, 0.95 * inch, 0.95 * inch, 0.8 * inch, 2.65 * inch])
    t_dis.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#0F2942')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.white, colors.HexColor('#F8FAFC')]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#FEF3C7')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    story.append(t_dis)
    story.append(Spacer(1, 8))

    # Section 4: Key Insights
    story.append(Paragraph("4. Key Technical Insights", h1_style))
    insights = [
        "<b>Input Resolution ($384\times 384$):</b> Tripled pixel density compared to standard $224\times 224$, preserving punctate lesions like microaneurysms and neuroretinal rim transitions.",
        "<b>Luminance Equalization (CLAHE):</b> Corrected peripheral illumination fall-off in the LAB color space, stabilizing feature extraction around vascular arcades.",
        "<b>Spatial Attention (CBAM):</b> Directly improved discrimination between Glaucoma and Disc Edema by weighting optic disc features.",
        "<b>Ensemble Synergy:</b> Combining high-resolution CNN representations with multimodal vision-language foundation model embeddings reduced classification variance.",
    ]
    for ins in insights:
        story.append(Paragraph(f"• {ins}", bullet_style))

    doc.build(story)
    print(f"Publication PDF successfully generated: {filename}")


if __name__ == "__main__":
    build_pdf()
