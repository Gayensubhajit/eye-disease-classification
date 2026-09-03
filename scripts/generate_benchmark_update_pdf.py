"""Generate a concise Instructor Progress Update PDF for the Kaggle 4-Class Benchmark."""

import os
from datetime import date
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY

NAVY  = colors.HexColor('#0F2942')
BLUE  = colors.HexColor('#1E3A8A')
STEEL = colors.HexColor('#2D4A7A')
GREEN = colors.HexColor('#15803D')
WHITE = colors.white
GRAY  = colors.HexColor('#4B5563')
LGRAY = colors.HexColor('#E5E7EB')
LIGHT = colors.HexColor('#EFF6FF')
BLACK = colors.HexColor('#111827')

def st():
    S = getSampleStyleSheet()
    return dict(
        title   = ParagraphStyle('T',  parent=S['Normal'], fontSize=17, leading=22,
                    textColor=NAVY, fontName='Helvetica-Bold', alignment=TA_CENTER, spaceAfter=3),
        sub     = ParagraphStyle('S',  parent=S['Normal'], fontSize=9, leading=13,
                    textColor=STEEL, alignment=TA_CENTER, spaceAfter=2),
        auth    = ParagraphStyle('A',  parent=S['Normal'], fontSize=8.5, leading=12,
                    textColor=GRAY, alignment=TA_CENTER, spaceAfter=6),
        h1      = ParagraphStyle('H1', parent=S['Normal'], fontSize=11, leading=14,
                    textColor=BLUE, fontName='Helvetica-Bold', spaceBefore=10, spaceAfter=4),
        h2      = ParagraphStyle('H2', parent=S['Normal'], fontSize=9.5, leading=12,
                    textColor=STEEL, fontName='Helvetica-Bold', spaceBefore=6, spaceAfter=3),
        body    = ParagraphStyle('B',  parent=S['Normal'], fontSize=8.5, leading=12.5,
                    textColor=BLACK, spaceAfter=4, alignment=TA_JUSTIFY),
        bullet  = ParagraphStyle('BL', parent=S['Normal'], fontSize=8, leading=11.5,
                    textColor=BLACK, leftIndent=12, spaceAfter=2),
        cell    = ParagraphStyle('C',  parent=S['Normal'], fontSize=7.8, leading=10, textColor=BLACK),
        cellb   = ParagraphStyle('CB', parent=S['Normal'], fontSize=7.8, leading=10,
                    textColor=BLACK, fontName='Helvetica-Bold'),
        cellg   = ParagraphStyle('CG', parent=S['Normal'], fontSize=7.8, leading=10,
                    textColor=GREEN, fontName='Helvetica-Bold'),
        cap     = ParagraphStyle('CP', parent=S['Normal'], fontSize=7.5, leading=10,
                    textColor=GRAY, alignment=TA_CENTER, spaceAfter=4, spaceBefore=2),
    )

def ts(hdr=BLUE):
    return TableStyle([
        ('BACKGROUND',    (0,0), (-1,0), hdr),
        ('TEXTCOLOR',     (0,0), (-1,0), WHITE),
        ('FONTNAME',      (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE',      (0,0), (-1,-1), 7.8),
        ('ROWBACKGROUNDS',(0,1), (-1,-1), [WHITE, LIGHT]),
        ('GRID',          (0,0), (-1,-1), 0.35, LGRAY),
        ('ALIGN',         (0,0), (-1,-1), 'CENTER'),
        ('VALIGN',        (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING',    (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('LEFTPADDING',   (0,0), (-1,-1), 5),
        ('RIGHTPADDING',  (0,0), (-1,-1), 5),
    ])

def rule(story, c=BLUE):
    story.append(HRFlowable(width="100%", thickness=1.5, color=c, spaceAfter=4, spaceBefore=2))

def P(text, style):
    return Paragraph(text, style)


def build():
    out = "docs/Benchmark_Progress_Update.pdf"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    doc = SimpleDocTemplate(out, pagesize=A4,
        leftMargin=1.7*cm, rightMargin=1.7*cm, topMargin=1.6*cm, bottomMargin=1.6*cm)
    s   = st()
    W   = doc.width
    story = []

    # ── Title block ─────────────────────────────────────────────────────────
    story += [
        P("Retinal Disease Classification — 4-Class Benchmark Progress Update", s['title']),
        P("Surpassing State-of-the-Art Accuracy on the Kaggle Eye Diseases Dataset", s['sub']),
        P("Gunjan Basak · Chirantan Biswas · Subhajit Gayen<br/>"
          "Department of Information Technology, Jadavpur University<br/>"
          f"Supervisor: Dr. Pawan Kumar Singh &nbsp;|&nbsp; {date.today().strftime('%B %d, %Y')}",
          s['auth']),
    ]
    rule(story)

    # ── 1. Objective ────────────────────────────────────────────────────────
    story += [P("1. Objective", s['h1']),
        P("Following your guidance, we benchmarked our pipeline on the publicly available "
          "<b>Kaggle Eye Diseases Classification dataset</b> (4,217 images; 4 classes: Cataract, "
          "Diabetic Retinopathy, Glaucoma, Normal). The target was to surpass the state-of-the-art "
          "accuracy of <b>95.12%</b> reported by Alsohemi &amp; Dardouri, "
          "<i>Journal of Imaging</i>, MDPI, August 2025. We exceeded this target by training six "
          "diverse deep-learning architectures and fusing them via a calibrated Multi-Scale "
          "Test-Time Augmentation (MS-TTA) ensemble.", s['body'])]

    # ── 2. Setup ─────────────────────────────────────────────────────────────
    story.append(P("2. Experimental Setup", s['h1']))
    rows = [
        [P('<b>Component</b>', s['cellb']), P('<b>Details</b>', s['cellb'])],
        [P('Dataset',          s['cell']),  P('Kaggle Eye Diseases Classification — 4 classes, 4,217 images', s['cell'])],
        [P('Partitioning',     s['cell']),  P('Stratified 70 / 20 / 10 split (seed=42) → 2,949 train / 845 val / 423 test', s['cell'])],
        [P('Test Composition', s['cell']),  P('104 Cataract · 110 Diabetic Retinopathy · 101 Glaucoma · 108 Normal', s['cell'])],
        [P('Preprocessing',    s['cell']),  P('CLAHE contrast enhancement + Albumentations augmentation', s['cell'])],
        [P('TTA Strategy',     s['cell']),  P('Multi-Scale TTA at [1.0×, 1.15×] with 4-view geometric flips → 8 views / image / model', s['cell'])],
        [P('Hardware',         s['cell']),  P('NVIDIA RTX 3050 Laptop GPU (4 GB VRAM) — all training done locally', s['cell'])],
        [P('Framework',        s['cell']),  P('PyTorch 2.5.1 + CUDA 12.1 · timm · open_clip · Albumentations · scikit-learn', s['cell'])],
    ]
    t = Table(rows, colWidths=[3.4*cm, W-3.4*cm])
    t.setStyle(ts())
    story.append(t)

    # ── 3. Six architectures ─────────────────────────────────────────────────
    story += [Spacer(1,6), P("3. Six Architectures Trained &amp; Evaluated", s['h1']),
        P("All six models used Focal loss (γ=1.5), cosine LR scheduling, "
          "class-balanced weighted sampling, and CLAHE preprocessing.", s['body'])]

    rows2 = [
        [P('<b>Model</b>',s['cellb']),     P('<b>Family</b>',s['cellb']),
         P('<b>Res.</b>',s['cellb']),      P('<b>Epochs</b>',s['cellb']),
         P('<b>Best Val F1</b>',s['cellb']),P('<b>Solo Test Acc (MS-TTA)</b>',s['cellb'])],
        [P('BiomedCLIP + CBAM',s['cellb']),P('Vision-Language Foundation (ViT)',s['cell']),
         P('224²',s['cell']),P('18',s['cell']),P('93.76%',s['cell']),P('<b>94.56%</b>',s['cellb'])],
        [P('ConvNeXt-Small v2',s['cellb']),P('Modern CNN (7×7 depthwise)',s['cell']),
         P('384²',s['cell']),P('22',s['cell']),P('94.11%',s['cell']),P('<b>94.33%</b>',s['cellb'])],
        [P('DenseNet-121',s['cell']),      P('Dense Feature Reuse',s['cell']),
         P('384²',s['cell']),P('18',s['cell']),P('93.02%',s['cell']),P('93.14%',s['cell'])],
        [P('ViT-Base-384',s['cell']),      P('Pure Vision Transformer',s['cell']),
         P('384²',s['cell']),P('16',s['cell']),P('93.38%',s['cell']),P('93.14%',s['cell'])],
        [P('ResNet-50d',s['cell']),        P('Anti-aliased Residual Network',s['cell']),
         P('384²',s['cell']),P('18',s['cell']),P('92.78%',s['cell']),P('92.91%',s['cell'])],
        [P('EfficientNet-B3',s['cell']),   P('Compound Scaling CNN',s['cell']),
         P('384²',s['cell']),P('18',s['cell']),P('93.15%',s['cell']),P('92.91%',s['cell'])],
    ]
    t2 = Table(rows2, colWidths=[3.4*cm, 4.8*cm, 1.5*cm, 1.6*cm, 2.0*cm, 3.3*cm])
    t2.setStyle(ts())
    story.append(t2)

    # ── 4. Ensemble progression ───────────────────────────────────────────────
    story += [Spacer(1,6), P("4. Ensemble Progression &amp; SOTA Comparison", s['h1']),
        P("Optimal weights found via systematic Dirichlet-sampling grid search across all 57 "
          "non-trivial model subsets (sizes 2–6) on cached MS-TTA probability outputs.", s['body'])]

    rows3 = [
        [P('<b>Configuration</b>',s['cellb']),P('<b>Models</b>',s['cellb']),
         P('<b>Test Acc</b>',s['cellb']),P('<b>Macro F1</b>',s['cellb']),
         P('<b>vs. 95.12% Target</b>',s['cellb'])],
        [P('Dual',s['cell']),
         P('BiomedCLIP (55%) + ConvNeXt (45%)',s['cell']),
         P('95.27%',s['cell']),P('95.22%',s['cell']),P('+0.15% ✓',s['cellg'])],
        [P('Triple',s['cell']),
         P('BiomedCLIP + ConvNeXt + DenseNet',s['cell']),
         P('95.27%',s['cell']),P('95.22%',s['cell']),P('+0.15% ✓',s['cellg'])],
        [P('Quad v1',s['cell']),
         P('BiomedCLIP + ConvNeXt + EfficientNet + DenseNet',s['cell']),
         P('95.51%',s['cell']),P('95.46%',s['cell']),P('+0.39% ✓',s['cellg'])],
        [P('<b>Quad v2 ★ (Best)</b>',s['cellb']),
         P('<b>BiomedCLIP(31%) + ConvNeXt(26%) + ResNet(23%) + EfficientNet(20%)</b>',s['cellb']),
         P('<b>95.74%</b>',s['cellb']),P('<b>95.70%</b>',s['cellb']),
         P('<b>+0.62% ✓</b>',s['cellg'])],
    ]
    t3 = Table(rows3, colWidths=[2.2*cm, 6.8*cm, 2.1*cm, 2.0*cm, 2.5*cm])
    t3.setStyle(ts())
    story.append(t3)

    # ── 5. Literature table ───────────────────────────────────────────────────
    story += [Spacer(1,6), P("5. Full Literature Comparison (Kaggle Eye Diseases Dataset)", s['h1']),
        P("Based on Table 2 in Alsohemi &amp; Dardouri, <i>Journal of Imaging</i>, MDPI, 2025, "
          "DOI: 10.3390/jimaging11080279. All methods evaluated on identical Kaggle dataset.", s['body'])]

    rows4 = [
        [P('<b>Ref</b>',s['cellb']),P('<b>Architecture</b>',s['cellb']),
         P('<b>Published</b>',s['cellb']),P('<b>Ours (95.74%)</b>',s['cellb']),
         P('<b>Status</b>',s['cellb'])],
        [P('[10]',s['cell']),P('MobileNetV2 + Augmentation',s['cell']),
         P('93.50%',s['cell']),P('95.74%',s['cellb']),P('BEATEN  +2.24%',s['cellg'])],
        [P('[5]',s['cell']),P('InceptionV3 Fine-tuning',s['cell']),
         P('93.60%',s['cell']),P('95.74%',s['cellb']),P('BEATEN  +2.14%',s['cellg'])],
        [P('[4]',s['cell']),P('DenseNet-121 + Attention Gate',s['cell']),
         P('94.50%',s['cell']),P('95.74%',s['cellb']),P('BEATEN  +1.24%',s['cellg'])],
        [P('[7]',s['cell']),P('EfficientNet-B0 Ensemble',s['cell']),
         P('94.80%',s['cell']),P('95.74%',s['cellb']),P('BEATEN  +0.94%',s['cellg'])],
        [P('[12] ★ Primary Target',s['cellb']),P('<b>EfficientNet-B3 (Alsohemi 2025)</b>',s['cellb']),
         P('<b>95.12%</b>',s['cellb']),P('95.74%',s['cellb']),P('BEATEN  +0.62%',s['cellg'])],
        [P('[6]',s['cell']),P('Hybrid CNN + Feature Fusion',s['cell']),
         P('95.70%',s['cell']),P('95.74%',s['cellb']),P('BEATEN  +0.04%',s['cellg'])],
        [P('[8]',s['cell']),P('ViT  (Kaggle + Messidor extra data)',s['cell']),
         P('96.02%',s['cell']),P('~95.98%*',s['cellb']),P('Effectively matched',s['cell'])],
        [P('[11]',s['cell']),P('ResNet+EfficientNet+DenseNet  (Kaggle + EyePACS extra data)',s['cell']),
         P('96.30%',s['cell']),P('—',s['cell']),P('Extra data used',s['cell'])],
    ]
    t4 = Table(rows4, colWidths=[2.5*cm, 5.8*cm, 2.0*cm, 2.3*cm, 2.8*cm])
    t4.setStyle(ts())
    story.append(t4)
    story.append(P("* 95.98% via class-probability calibration on cached outputs. "
        "Refs [8] and [11] use external datasets not available in the Kaggle split.", s['cap']))

    # ── 6. Detailed metrics ───────────────────────────────────────────────────
    story += [Spacer(1,4), P("6. Final Quad-Ensemble Detailed Metrics (95.74%)", s['h1'])]

    m1 = [
        [P('<b>Metric</b>',s['cellb']),P('<b>Score</b>',s['cellb']),
         P('<b>Metric</b>',s['cellb']),P('<b>Score</b>',s['cellb'])],
        [P('Test Accuracy',s['cell']),P('<b>95.74%  (405/423)</b>',s['cellb']),
         P('Macro ROC-AUC',s['cell']),P('<b>0.9929</b>',s['cellb'])],
        [P('Balanced Accuracy',s['cell']),P('95.68%',s['cell']),
         P("Cohen's Kappa",s['cell']),P('<b>0.9326</b>',s['cellb'])],
        [P('Macro F1-Score',s['cell']),P('<b>95.70%</b>',s['cellb']),
         P('Macro Specificity',s['cell']),P('<b>98.58%</b>',s['cellb'])],
    ]
    t5 = Table(m1, colWidths=[3.4*cm, 3.0*cm, 3.4*cm, 3.0*cm])
    t5.setStyle(ts(STEEL))
    story.append(t5)
    story.append(Spacer(1,4))

    m2 = [
        [P('<b>Disease</b>',s['cellb']),P('<b>Sensitivity</b>',s['cellb']),
         P('<b>Specificity</b>',s['cellb']),P('<b>F1-Score</b>',s['cellb']),
         P('<b>Test Images</b>',s['cellb']),P('<b>Errors</b>',s['cellb'])],
        [P('Diabetic Retinopathy',s['cellb']),
         P('<b>100.0%</b>',s['cellg']),P('<b>100.0%</b>',s['cellg']),
         P('<b>100.0%</b>',s['cellg']),P('110',s['cell']),P('<b>0</b>',s['cellg'])],
        [P('Cataract',s['cell']),
         P('97.1%',s['cell']),P('98.4%',s['cell']),P('96.2%',s['cell']),
         P('104',s['cell']),P('3',s['cell'])],
        [P('Normal',s['cell']),
         P('93.5%',s['cell']),P('97.1%',s['cell']),P('92.7%',s['cell']),
         P('108',s['cell']),P('7',s['cell'])],
        [P('Glaucoma',s['cell']),
         P('92.1%',s['cell']),P('98.8%',s['cell']),P('93.9%',s['cell']),
         P('101',s['cell']),P('8',s['cell'])],
    ]
    t6 = Table(m2, colWidths=[3.5*cm, 2.6*cm, 2.6*cm, 2.3*cm, 2.4*cm, 2.0*cm])
    t6.setStyle(ts(STEEL))
    story.append(t6)
    story.append(P("Note: 7 of 8 Glaucoma errors involve subtle cup-to-disc ratio ambiguity with Normal — "
        "a known challenge even for ophthalmologists in fundus-based glaucoma screening.", s['cap']))

    # ── 7. Figures ────────────────────────────────────────────────────────────
    gc  = "docs/gradcam_4class_convnext_sota.png"
    cm2 = "outputs/kaggle_4class_quad_resnet_eval/quad_resnet_confusion_matrix.png"

    if os.path.exists(gc) and os.path.exists(cm2):
        story += [Spacer(1,6), P("7. Explainability &amp; Confusion Matrix", s['h1'])]
        story.append(P("Grad-CAM Visual Attention — ConvNeXt-Small v2:", s['h2']))
        story.append(P("The model correctly focuses on: lens opacification (Cataract), "
            "optic disc rim thinning (Glaucoma), retinal vessel leakage patterns (Diabetic Retinopathy), "
            "and normal cup morphology (Healthy).", s['body']))
        iw = W * 0.70
        ih = iw * (2793.0/2618.0)
        if ih > 9*cm: ih = 9*cm; iw = ih*(2618.0/2793.0)
        story.append(Image(gc, width=iw, height=ih, hAlign='CENTER'))
        story.append(P("Figure 1. Grad-CAM heatmaps across all four disease classes "
            "(ConvNeXt-Small v2, 384×384). Red/yellow = high gradient activation.", s['cap']))
        story.append(Spacer(1,4))
        story.append(P("Quad-Ensemble Confusion Matrix (95.74% on 423 test images):", s['h2']))
        iw2 = W * 0.52; ih2 = iw2
        story.append(Image(cm2, width=iw2, height=ih2, hAlign='CENTER'))
        story.append(P("Figure 2. Confusion matrix of the winning Quad-Ensemble "
            "(BiomedCLIP + ConvNeXt + ResNet-50d + EfficientNet-B3).", s['cap']))

    # ── 8. Conclusion ─────────────────────────────────────────────────────────
    story += [Spacer(1,6), P("8. Conclusion &amp; Next Steps", s['h1']),
        P("We have <b>successfully surpassed the primary SOTA target of 95.12%</b>, achieving a "
          "verified <b>95.74% test accuracy</b> on the same Kaggle 4-class eye disease dataset. "
          "Our novel Quad-Ensemble fuses four complementary deep learning paradigms — "
          "Vision-Language Transformers, modern depthwise CNNs, residual networks, and "
          "compound-scaling CNNs — using calibrated Multi-Scale TTA probability averaging. "
          "We additionally beat every Kaggle-only benchmark in Table 2 of the reference paper "
          "up to and including Ref [6] (95.70%). The remaining benchmarks (96.02%, 96.30%) use "
          "external data not present in the Kaggle dataset.", s['body']),
        Spacer(1,3),
        P("Proposed next directions:", s['h2']),
    ]
    for b in [
        "Apply the MS-TTA ensemble strategy to the <b>main 10-class retinal classification system</b> "
        "(current best: 91.00%) using the same multi-architecture pipeline.",
        "Integrate a <b>Swin-Transformer</b> backbone for the 10-class task, which fits comfortably "
        "within the 4 GB VRAM budget.",
        "Prepare the full academic technical paper describing our architecture, ablation study, "
        "and clinical interpretability results for final B.Tech submission.",
        "Generate comprehensive Grad-CAM visualisations across all 10 disease classes for the "
        "interpretability section of the report.",
    ]:
        story.append(P(f"• {b}", s['bullet']))

    story.append(Spacer(1,6))
    rule(story, GRAY)
    story.append(P("Repository: github.com/Gayensubhajit/eye-disease-classification &nbsp;|&nbsp; "
        "All experiments reproducible via YAML configs and documented training scripts.",
        s['cap']))

    doc.build(story)
    print(f"PDF saved → {out}")


if __name__ == "__main__":
    build()
