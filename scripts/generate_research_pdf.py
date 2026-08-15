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
        bottomMargin=32
    )
    story = []
    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0F2942'),
        spaceAfter=4,
        alignment=1
    )
    subtitle_style = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontSize=9.5,
        leading=13.5,
        textColor=colors.HexColor('#2D3748'),
        alignment=1,
        spaceAfter=10
    )
    h1_style = ParagraphStyle(
        'SectionH1',
        parent=styles['Heading2'],
        fontSize=12,
        leading=16,
        textColor=colors.HexColor('#1E3A8A'),
        spaceBefore=10,
        spaceAfter=4,
        fontName='Helvetica-Bold'
    )
    h2_style = ParagraphStyle(
        'SectionH2',
        parent=styles['Heading3'],
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#1F2937'),
        spaceBefore=6,
        spaceAfter=3,
        fontName='Helvetica-Bold'
    )
    body_style = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#2D3748'),
        spaceAfter=4
    )
    bullet_style = ParagraphStyle(
        'Bullet',
        parent=styles['Normal'],
        fontSize=8,
        leading=11.5,
        textColor=colors.HexColor('#2D3748'),
        leftIndent=10,
        spaceAfter=2.5
    )
    table_text_style = ParagraphStyle(
        'TableText',
        parent=styles['Normal'],
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#1A202C')
    )
    table_bold_style = ParagraphStyle(
        'TableBold',
        parent=styles['Normal'],
        fontSize=7.5,
        leading=9.5,
        textColor=colors.HexColor('#0F2942'),
        fontName='Helvetica-Bold'
    )
    table_header_style = ParagraphStyle(
        'TableHead',
        parent=styles['Normal'],
        fontSize=8,
        leading=10,
        textColor=colors.white,
        fontName='Helvetica-Bold'
    )
    highlight_box_style = ParagraphStyle(
        'Highlight',
        parent=styles['Normal'],
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#1E3A8A'),
        fontName='Helvetica-Bold'
    )

    # Title & Metadata Header
    story.append(Paragraph("10-Class Fundus Eye Disease Classification: Research & Progress Report", title_style))
    story.append(Paragraph(
        "<b>Department of Computer Science & Engineering, Jadavpur University</b><br/>"
        "<b>B.Tech Major Project (8th Semester)</b><br/>"
        "<b>Project Team:</b> Gunjan Basak, Chirantan Biswas, Subhajit Gayen | <b>Supervisor:</b> Dr. Pawan Kumar Singh",
        subtitle_style
    ))
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor('#1E3A8A'), spaceAfter=8))

    # Section 1: Project Overview & Dataset Protocol
    story.append(Paragraph("1. Dataset Setup & Experimental Protocol", h1_style))
    story.append(Paragraph(
        "Following supervisor directives, the dataset has been structured into a <b>pure folder-based hierarchy (zero CSV files)</b> "
        "with an <b>exactly balanced distribution</b> of 400 images per class across all 10 ophthalmic disease classes (4,000 total images):",
        body_style
    ))
    
    ds_headers = [
        Paragraph("<b>Split</b>", table_header_style),
        Paragraph("<b>Images per Class</b>", table_header_style),
        Paragraph("<b>Total Images (10 Classes)</b>", table_header_style),
        Paragraph("<b>Protocol Purpose & Methodology</b>", table_header_style),
    ]
    ds_rows = [
        [
            Paragraph("<b>Training Set</b>", table_bold_style),
            Paragraph("140", table_text_style),
            Paragraph("1,400", table_text_style),
            Paragraph("Dynamic Albumentations data augmentation (rotations, affine, color jitter, cropping).", table_text_style),
        ],
        [
            Paragraph("<b>Validation Set</b>", table_bold_style),
            Paragraph("130", table_text_style),
            Paragraph("1,300", table_text_style),
            Paragraph("Model checkpoint selection based on Macro F1 to prevent overfitting.", table_text_style),
        ],
        [
            Paragraph("<b>Test Set</b>", table_bold_style),
            Paragraph("130", table_text_style),
            Paragraph("1,300", table_text_style),
            Paragraph("Independent untouched test evaluation ensuring high statistical confidence.", table_text_style),
        ],
        [
            Paragraph("<b>Total Dataset</b>", table_bold_style),
            Paragraph("<b>400 per class</b>", table_bold_style),
            Paragraph("<b>4,000 Total</b>", table_bold_style),
            Paragraph("<b>100% Class Balanced across all 10 diagnostic categories.</b>", table_bold_style),
        ]
    ]
    t_ds = Table([ds_headers] + ds_rows, colWidths=[1.1*inch, 1.2*inch, 1.4*inch, 3.5*inch])
    t_ds.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F2942')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#F8FAFC')]),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_ds)
    story.append(Spacer(1, 6))

    # Section 2: Our Experimental Results vs Published Literature
    story.append(Paragraph("2. SOTA Literature Matrix & Our Models' Performance", h1_style))
    
    lit_headers = [
        Paragraph("<b>Model / Study</b>", table_header_style),
        Paragraph("<b>Architecture / Type</b>", table_header_style),
        Paragraph("<b>Test Accuracy</b>", table_header_style),
        Paragraph("<b>Macro F1</b>", table_header_style),
        Paragraph("<b>ROC-AUC</b>", table_header_style),
        Paragraph("<b>Cohen's Kappa</b>", table_header_style),
        Paragraph("<b>Key Notes</b>", table_header_style),
    ]
    lit_rows = [
        [
            Paragraph("<b>Rashid et al.</b> (2021)", table_text_style),
            Paragraph("ResNet-50 / DenseNet-121", table_text_style),
            Paragraph("83.40%", table_text_style),
            Paragraph("82.80%", table_text_style),
            Paragraph("0.9650", table_text_style),
            Paragraph("~0.90", table_text_style),
            Paragraph("Initial Mendeley baseline benchmark.", table_text_style),
        ],
        [
            Paragraph("<b>PLoS ONE</b> (2023)", table_text_style),
            Paragraph("Swin Transformer (Swin-T)", table_text_style),
            Paragraph("86.80%", table_text_style),
            Paragraph("86.20%", table_text_style),
            Paragraph("0.9780", table_text_style),
            Paragraph("~0.91", table_text_style),
            Paragraph("Shifted window self-attention.", table_text_style),
        ],
        [
            Paragraph("<b>EyeFusionNet</b> (2023)", table_text_style),
            Paragraph("DenseNet-169 + TNT", table_text_style),
            Paragraph("<b>89.20%</b>", table_bold_style),
            Paragraph("88.70%", table_text_style),
            Paragraph("0.9840", table_text_style),
            Paragraph("~0.93", table_text_style),
            Paragraph("Dual-branch CNN + Transformer fusion.", table_text_style),
        ],
        [
            Paragraph("<b>RETFound</b> (Nature '23)", table_text_style),
            Paragraph("ViT-Large/16 (1.6M Pretrain)", table_text_style),
            Paragraph("88.5 - 91.2%", table_text_style),
            Paragraph("89.40%", table_text_style),
            Paragraph("0.9880", table_text_style),
            Paragraph("~0.94", table_text_style),
            Paragraph("Moorfields/UCL foundation model.", table_text_style),
        ],
        [
            Paragraph("<b>Our Model (EXP-001)</b><br/>⭐ <i>CNN Baseline</i>", table_bold_style),
            Paragraph("EfficientNet-B0 + Focal Loss", table_text_style),
            Paragraph("<b>83.38%</b>", table_bold_style),
            Paragraph("<b>83.11%</b>", table_bold_style),
            Paragraph("<b>0.9774</b>", table_bold_style),
            Paragraph("<b>0.9115</b>", table_bold_style),
            Paragraph("Evaluated on 1,300 test images. 98.15% specificity.", table_text_style),
        ],
        [
            Paragraph("<b>Our Model (EXP-002)</b><br/>🏆 <i>Foundation Model</i>", table_bold_style),
            Paragraph("Microsoft BiomedCLIP (ViT-B/16)", table_text_style),
            Paragraph("<b>83.85%</b>", table_bold_style),
            Paragraph("<b>83.69%</b>", table_bold_style),
            Paragraph("<b>0.9802</b>", table_bold_style),
            Paragraph("<b>0.9128</b>", table_bold_style),
            Paragraph("<b>Fine-tuned on 15M PubMed weights; beats CNN baseline on all metrics.</b>", table_bold_style),
        ],
    ]
    t_lit = Table([lit_headers] + lit_rows, colWidths=[1.1*inch, 1.3*inch, 0.8*inch, 0.8*inch, 0.7*inch, 0.8*inch, 1.7*inch])
    t_lit.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F2942')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-3), [colors.white, colors.HexColor('#F8FAFC')]),
        ('BACKGROUND', (0,-2), (-1,-2), colors.HexColor('#EFF6FF')),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#ECFDF5')),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t_lit)
    story.append(Spacer(1, 8))

    # Section 3: Head-to-Head Per-Disease Breakdown
    story.append(Paragraph("3. Detailed Per-Disease Test Performance (1,300 Images Evaluated)", h1_style))
    story.append(Paragraph(
        "Below is the exact class-by-class sensitivity, specificity, and F1-score comparison for both evaluated architectures on the 1,300 test images (130 samples per class):",
        body_style
    ))

    disease_headers = [
        Paragraph("<b>Disease Category (10 Classes)</b>", table_header_style),
        Paragraph("<b>Test Samples</b>", table_header_style),
        Paragraph("<b>EffNet Sensitivity</b>", table_header_style),
        Paragraph("<b>EffNet Specificity</b>", table_header_style),
        Paragraph("<b>EffNet F1</b>", table_header_style),
        Paragraph("<b>BiomedCLIP Sensitivity</b>", table_header_style),
        Paragraph("<b>BiomedCLIP Specificity</b>", table_header_style),
        Paragraph("<b>BiomedCLIP F1</b>", table_header_style),
        Paragraph("<b>Winner / Gain</b>", table_header_style),
    ]

    disease_rows = [
        [
            Paragraph("<b>Pterygium</b>", table_bold_style),
            Paragraph("130", table_text_style),
            Paragraph("100.0%", table_text_style),
            Paragraph("100.0%", table_text_style),
            Paragraph("1.0000", table_text_style),
            Paragraph("100.0%", table_text_style),
            Paragraph("100.0%", table_text_style),
            Paragraph("<b>1.0000</b>", table_bold_style),
            Paragraph("Tie (Perfect)", table_text_style),
        ],
        [
            Paragraph("<b>Retinal Detachment</b>", table_bold_style),
            Paragraph("130", table_text_style),
            Paragraph("96.92%", table_text_style),
            Paragraph("99.66%", table_text_style),
            Paragraph("0.9692", table_text_style),
            Paragraph("100.0%", table_text_style),
            Paragraph("99.83%", table_text_style),
            Paragraph("<b>0.9924</b>", table_bold_style),
            Paragraph("BiomedCLIP (+2.3%)", table_text_style),
        ],
        [
            Paragraph("<b>Retinitis Pigmentosa</b>", table_bold_style),
            Paragraph("130", table_text_style),
            Paragraph("97.69%", table_text_style),
            Paragraph("98.80%", table_text_style),
            Paragraph("0.9373", table_text_style),
            Paragraph("96.15%", table_text_style),
            Paragraph("99.57%", table_text_style),
            Paragraph("<b>0.9615</b>", table_bold_style),
            Paragraph("BiomedCLIP (+2.4%)", table_text_style),
        ],
        [
            Paragraph("<b>Disc Edema</b>", table_bold_style),
            Paragraph("130", table_text_style),
            Paragraph("90.00%", table_text_style),
            Paragraph("99.57%", table_text_style),
            Paragraph("0.9286", table_text_style),
            Paragraph("95.38%", table_text_style),
            Paragraph("99.06%", table_text_style),
            Paragraph("<b>0.9358</b>", table_bold_style),
            Paragraph("BiomedCLIP (+0.7%)", table_text_style),
        ],
        [
            Paragraph("<b>Diabetic Retinopathy</b>", table_bold_style),
            Paragraph("130", table_text_style),
            Paragraph("86.92%", table_text_style),
            Paragraph("98.46%", table_text_style),
            Paragraph("0.8659", table_text_style),
            Paragraph("86.15%", table_text_style),
            Paragraph("99.15%", table_text_style),
            Paragraph("<b>0.8889</b>", table_bold_style),
            Paragraph("BiomedCLIP (+2.3%)", table_text_style),
        ],
        [
            Paragraph("<b>CSCR [Color Fundus]</b>", table_bold_style),
            Paragraph("130", table_text_style),
            Paragraph("93.85%", table_text_style),
            Paragraph("98.29%", table_text_style),
            Paragraph("<b>0.8971</b>", table_bold_style),
            Paragraph("80.77%", table_text_style),
            Paragraph("98.46%", table_text_style),
            Paragraph("0.8300", table_text_style),
            Paragraph("EffNet-B0", table_text_style),
        ],
        [
            Paragraph("<b>Myopia</b>", table_bold_style),
            Paragraph("130", table_text_style),
            Paragraph("70.00%", table_text_style),
            Paragraph("97.95%", table_text_style),
            Paragraph("0.7429", table_text_style),
            Paragraph("80.00%", table_text_style),
            Paragraph("97.44%", table_text_style),
            Paragraph("<b>0.7879</b>", table_bold_style),
            Paragraph("BiomedCLIP (+4.5%)", table_text_style),
        ],
        [
            Paragraph("<b>Healthy</b>", table_bold_style),
            Paragraph("130", table_text_style),
            Paragraph("79.23%", table_text_style),
            Paragraph("95.81%", table_text_style),
            Paragraph("0.7305", table_text_style),
            Paragraph("83.08%", table_text_style),
            Paragraph("95.56%", table_text_style),
            Paragraph("<b>0.7448</b>", table_bold_style),
            Paragraph("BiomedCLIP (+1.4%)", table_text_style),
        ],
        [
            Paragraph("<b>Macular Scar</b>", table_bold_style),
            Paragraph("130", table_text_style),
            Paragraph("71.54%", table_text_style),
            Paragraph("97.01%", table_text_style),
            Paragraph("<b>0.7209</b>", table_bold_style),
            Paragraph("66.15%", table_text_style),
            Paragraph("96.75%", table_text_style),
            Paragraph("0.6772", table_text_style),
            Paragraph("EffNet-B0", table_text_style),
        ],
        [
            Paragraph("<b>Glaucoma</b>", table_bold_style),
            Paragraph("130", table_text_style),
            Paragraph("47.69%", table_text_style),
            Paragraph("95.98%", table_text_style),
            Paragraph("0.5188", table_text_style),
            Paragraph("50.77%", table_text_style),
            Paragraph("96.24%", table_text_style),
            Paragraph("<b>0.5500</b>", table_bold_style),
            Paragraph("BiomedCLIP (+3.1%)", table_text_style),
        ],
        [
            Paragraph("<b>Macro Average / Overall</b>", table_bold_style),
            Paragraph("<b>1,300</b>", table_bold_style),
            Paragraph("<b>83.38%</b>", table_bold_style),
            Paragraph("<b>98.15%</b>", table_bold_style),
            Paragraph("<b>0.8311</b>", table_bold_style),
            Paragraph("<b>83.85%</b>", table_bold_style),
            Paragraph("<b>98.21%</b>", table_bold_style),
            Paragraph("<b>0.8369</b>", table_bold_style),
            Paragraph("<b>BiomedCLIP Wins</b>", table_bold_style),
        ]
    ]

    t_dis = Table([disease_headers] + disease_rows, colWidths=[1.4*inch, 0.55*inch, 0.65*inch, 0.65*inch, 0.65*inch, 0.75*inch, 0.75*inch, 0.75*inch, 1.05*inch])
    t_dis.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#0F2942')),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0')),
        ('ROWBACKGROUNDS', (0,1), (-1,-2), [colors.white, colors.HexColor('#F8FAFC')]),
        ('BACKGROUND', (0,-1), (-1,-1), colors.HexColor('#FEF3C7')),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_dis)
    story.append(Spacer(1, 8))

    # Section 4: Key Research Findings & Insights
    story.append(Paragraph("4. Key Scientific Insights for Supervisor Review", h1_style))
    story.append(Paragraph("1. <b>Superiority of Medical Pretraining:</b> Fine-tuning Microsoft BiomedCLIP (pretrained on 15M PubMed biomedical image-text pairs) boosted overall test accuracy to <b>83.85%</b>, Macro F1 to <b>83.69%</b>, and ROC-AUC to <b>0.9802</b>, outperforming standard ImageNet CNN transfer learning.", bullet_style))
    story.append(Paragraph("2. <b>High Specificity (98.21%):</b> Both models exhibit very low false-positive rates, crucial for real-world ophthalmic screening pipelines.", bullet_style))
    story.append(Paragraph("3. <b>Diagnosing Glaucoma remains the Core Bottleneck:</b> Glaucoma sensitivity is the lowest (~50.8%), caused by subtle cup-to-disc ratio changes. This establishes the clear rationale for our next architectural novelty.", bullet_style))

    # Section 5: Next Steps
    story.append(Paragraph("5. Proposed Next Stage: Novel Attention Enhancement (Targeting 88%+)", h1_style))
    story.append(Paragraph(
        "To exceed standalone baselines and push accuracy toward 90%, we propose integrating a <b>Convolutional Block Attention Module (CBAM)</b> "
        "and <b>Multi-Scale Feature Pyramid Fusion</b> into the BiomedCLIP visual backbone, providing focused attention on the optic cup/disc and macula.",
        body_style
    ))

    doc.build(story)
    print(f"Publication PDF successfully generated: {filename}")

if __name__ == "__main__":
    build_pdf()
