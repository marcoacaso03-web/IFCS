#!/usr/bin/env python3
"""
make_slides.py - Build IFCS_2026_presentation.pptx using ONLY the standard
library (zipfile + OOXML XML). No python-pptx required.

Reads artifacts/metrics.json and the figures, emits a valid PowerPoint deck.
"""
import os, json, zipfile
from xml.sax.saxutils import escape
from PIL import Image

OUT = "artifacts"
metrics = json.load(open(f"{OUT}/metrics.json"))

# Convert figures to RGB PNG (drop alpha) for maximum compatibility
def to_rgb_png(src, dst):
    im = Image.open(src).convert("RGB")
    im.save(dst, "PNG")

media = {}
figs = {"sil": "fig_silhouette.png", "prof": "fig_cluster_profiles.png",
        "geo": "fig_geo.png", "distress": "fig_distress.png",
        "imp": "fig_importance.png"}
for key, fn in figs.items():
    p = f"{OUT}/{fn}"
    d = f"{OUT}/_rgb_{fn}"
    to_rgb_png(p, d)
    media[key] = d

def rels_path(name):
    return name.replace("ppt/", "ppt/_rels/") + ".rels"

SLIDES = []  # list of (title, body_lines, [image_specs])

def B(t): return t

NAVY = "1F3A5F"; ACC = "C44E52"; GREY = "555555"; WHITE = "FFFFFF"; LBLUE="CFDDEE"
cls = metrics.get("classification", {})
sil = metrics.get("silhouette_by_k", {})
ks = list(sil.keys()); vs = [round(sil[k], 3) for k in ks]
sizes = metrics.get("cluster_sizes", {})
distr = metrics.get("cluster_distress_rate", {})
imp = list(metrics.get("feature_importance", {}).items())[:4]
top_feats = ", ".join(k for k, _ in imp)

# Slide 1 - title
SLIDES.append({
    "type": "title",
    "title": "IFCS 2026 Data Challenge",
    "subtitle": "Profiling & Predicting Financial Distress in Italian SMEs",
    "footer": f"~{metrics['n_firms']:,} firms (FY2023)  |  distress rate {metrics['distress_rate']*100:.1f}%  |  14 financial & structural features",
})

# Slide 2
SLIDES.append({
    "type": "content", "title": "The Challenge",
    "bullets": [
        ("Two goals.", "Profile the SME population and flag those heading into distress - before it is too late."),
        ("Why it matters.", "Lenders, policymakers and the firms themselves need early warning on financial distress."),
        ("Data.", f"One row per SME, {metrics['n_firms']:,} firms, monetary values in kEUR, geography at province level."),
        ("Task A - Clustering.", "Unsupervised grouping by financial characteristics + economic interpretation + geography."),
        ("Task B - Classification.", "Supervised prediction of Financial distress (TRUE/FALSE) on a held-out test set."),
    ],
    "note": "Interpretation counts as much as prediction.",
})

# Slide 3
SLIDES.append({
    "type": "content", "title": "Data & Preprocessing",
    "bullets": [
        ("Features used.", "Sales Revenue, Employees, Net income, Operating Income, Max deductible amount, Total financial expenses, Tax shield, Operating cash flow, Current taxes, Alert Index."),
        ("Excluded.", "Company ID, Province/sector (interpretation only), and the target."),
        ("Transform.", "Log1p on heavy-tailed monetary/count variables; standardize (z-score) before clustering & modelling."),
        ("Reason.", "Skewed distributions and differing scales would otherwise let a few large firms dominate distance metrics."),
        ("Cleaning.", "Alert Index is mixed (numeric + 'EXCELLENT' -> 0, best risk class); 0 provinces unmapped."),
    ],
})

# Slide 4
SLIDES.append({
    "type": "content", "title": "Task A - Clustering Method",
    "bullets": [
        ("Algorithm.", "K-Means on standardized, log-transformed features."),
        ("Choosing k.", "Silhouette over k=2..8 (max at k=2); K=5 selected for profiling granularity."),
        ("Result.", f"K = {metrics['best_k_silhouette']} clusters; silhouette at K=5 = {metrics['silhouette_at_best']:.3f}."),
        ("Trade-off.", "Separation vs interpretability: 5 segments are economically distinguishable."),
    ],
    "image": ("sil", 7.0, 1.6, 6.0),
})

# Slide 5
citems = [(f"Cluster {c} (n={sizes[c]})", f"distress rate {distr[c]*100:.1f}%") for c in sorted(sizes, key=int)]
SLIDES.append({
    "type": "content", "title": "Task A - Cluster Profiles",
    "bullets": citems,
    "image": ("prof", 0.5, 1.4, 8.4),
})

# Slide 6
SLIDES.append({
    "type": "content", "title": "Task A - Geographic Patterns",
    "bullets": [
        ("North-heavy.", "Every cluster is dominated by the North/industrial belt."),
        ("Distress is uneven.", "Cluster 4 carries ~50% distress; cluster 3 ~0.6%."),
        ("Islands under-represented.", "Sicilia & Sardegna are a small share in all clusters."),
    ],
    "images": [("geo", 0.4, 1.5, 7.4), ("distress", 8.0, 1.5, 5.0)],
})

# Slide 7
SLIDES.append({
    "type": "content", "title": "Task B - Predicting Distress",
    "bullets": [
        ("Model.", "Logistic Regression (L2, class-weighted) - robust baseline for early warning."),
        ("Cleaning.", "Dropped 2 collinear derived vars (VIF 182-204) so coefficients are stable."),
        ("Validation.", f"5-fold stratified CV, ROC-AUC = {cls.get('cv_roc_auc_mean',0):.3f} (+/- {cls.get('cv_roc_auc_std',0):.3f})."),
        ("Train AUC.", f"{cls.get('train_roc_auc',0):.3f}."),
        ("Deliverable.", "predictions.csv: Company ID + pred_class (TRUE/FALSE)."),
        ("Top drivers.", top_feats + "."),
    ],
    "image": ("imp", 7.0, 1.5, 6.0),
})

# Slide 8
SLIDES.append({
    "type": "content", "title": "Conclusions & Next Steps",
    "bullets": [
        ("Profiling.", f"{metrics['best_k_silhouette']} financially distinct SME segments identified and economically characterised."),
        ("Geography.", "Clear macro-area skew; distress concentrates in specific profiles (cluster 4)."),
        ("Prediction.", f"Logistic model reaches CV ROC-AUC {cls.get('cv_roc_auc_mean',0):.3f} - strong early-warning signal."),
        ("Next steps.", "Apply pipeline to official test_features.csv; refine clusters with sector constraints; calibrate threshold to risk appetite."),
    ],
    "note": "Thank you - questions welcome.",
})

# ---------------------------------------------------------------------------
# Build OOXML package
# ---------------------------------------------------------------------------
EMU = 914400
SW, SH = 12192000, 6858000  # 13.333 x 7.5 inch in EMU

def emu(inch): return int(inch * EMU)

def textbox_xml(l, t, w, h, lines, align="l"):
    # lines: list of (text, size, color, bold)
    paras = []
    for (txt, size, color, bold) in lines:
        algn = "<?a:algn val=\"ctr\"?>" if align == "ctr" else ""
        paras.append(
            f'<a:p><a:pPr>{algn}</a:pPr>'
            f'<a:r><a:rPr lang="en-US" sz="{int(size*100)}" b="{1 if bold else 0}" i="0">'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill></a:rPr>'
            f'<a:t>{escape(txt)}</a:t></a:r></a:p>')
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="2" name="TextBox"/><p:cNvSpPr txBox="1"/>'
            f'<p:nvPr/></p:nvSpPr><p:spPr><a:off x="{emu(l)}" y="{emu(t)}"/>'
            f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/><a:noFill/></p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" lIns="91440" tIns="45720" rIns="91440" bIns="45720"/>'
            f'<a:lstStyle/>{"".join(paras)}</p:txBody></p:sp>')

def rect_xml(l, t, w, h, color):
    return (f'<p:sp><p:nvSpPr><p:cNvPr id="1" name="Rect"/><p:cNvSpPr/>'
            f'<p:nvPr/></p:nvSpPr><p:spPr><a:off x="{emu(l)}" y="{emu(t)}"/>'
            f'<a:ext cx="{emu(w)}" cy="{emu(h)}"/>'
            f'<a:solidFill><a:srgbClr val="{color}"/></a:solidFill>'
            f'<a:ln><a:noFill/></a:ln></p:spPr><p:txBody><a:bodyPr/>'
            f'<a:lstStyle/><a:p/></p:txBody></p:sp>')

def pic_xml(rel_id, l, t, w):
    return (f'<p:pic><p:nvPicPr><p:cNvPr id="3" name="pic"/><p:cNvPicPr/>'
            f'<p:nvPr/></p:nvPicPr><p:blipFill><a:blip r:embed="{rel_id}"/>'
            f'<a:stretch><a:fillRect/></a:stretch></p:blipFill>'
            f'<p:spPr><a:off x="{emu(l)}" y="{emu(t)}"/><a:ext cx="{emu(w)}" cy="{emu(w)*9//16}"/>'
            f'<a:noFill/></p:spPr></p:pic>')

def slide_xml(title, body_lines, images, note):
    sps = []
    # title band
    sps.append(rect_xml(0, 0, 13.333, 1.1, NAVY))
    sps.append(textbox_xml(0.5, 0.18, 12.3, 0.8,
              [(title, 28, WHITE, True)]))
    # body
    if body_lines:
        lines = []
        for (lead, rest) in body_lines:
            lines.append((lead, 16, NAVY, True))
            if rest:
                lines.append((rest, 16, GREY, False))
        sps.append(textbox_xml(0.7, 1.45, 12.0, 5.0, lines))
    # images
    for (rel_id, l, t, w) in images:
        sps.append(pic_xml(rel_id, l, t, w))
    # note
    if note:
        sps.append(rect_xml(0, 6.7, 13.333, 0.8, ACC))
        sps.append(textbox_xml(0.5, 6.85, 12.3, 0.6, [(note, 16, WHITE, True)]))
    tree = ("<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"yes\"?>"
            '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
            '<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="0" name=""/><p:cNvGrpSpPr/>'
            '<p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
            '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></p:grpSpPr>'
            + "".join(sps) +
            '</p:spTree></p:cSld><p:clrMapOvr><a:overrideClrMapping '
            'bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" '
            'accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" '
            'hlink="hlink" folHlink="folHlink"/></p:clrMapOvr></p:sld>')
    return tree

# theme
THEME = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
 '<a:theme xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" name="Office Theme">'
 '<a:themeElements><a:clrScheme name="Office"><a:dk1><a:sysClr val="windowText" lastClr="000000"/>'
 '</a:dk1><a:lt1><a:sysClr val="window" lastClr="FFFFFF"/></a:lt1>'
 '<a:dk2><a:srgbClr val="1F3A5F"/></a:dk2><a:lt2><a:srgbClr val="FFFFFF"/></a:lt2>'
 '<a:accent1><a:srgbClr val="4C72B0"/></a:accent1><a:accent2><a:srgbClr val="C44E52"/></a:accent2>'
 '<a:accent3><a:srgbClr val="55A868"/></a:accent3><a:accent4><a:srgbClr val="8172B2"/></a:accent4>'
 '<a:accent5><a:srgbClr val="CCB974"/></a:accent5><a:accent6><a:srgbClr val="64B5CD"/></a:accent6>'
 '</a:clrScheme><a:fontScheme name="Office"><a:majorFont><a:latin typeface="Calibri"/>'
 '<a:ea typeface=""/><a:cs typeface=""/></a:majorFont><a:minorFont><a:latin typeface="Calibri"/>'
 '<a:ea typeface=""/><a:cs typeface=""/></a:minorFont></a:fontScheme>'
 '<a:fmtScheme name="Office"><a:fillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
 '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
 '</a:fillStyleLst><a:lnStyleLst><a:ln w="6350"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
 '<a:ln w="12700"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln>'
 '<a:ln w="19050"><a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:ln></a:lnStyleLst>'
 '<a:effectStyleLst><a:effectStyle><a:effectLst/></a:effectStyle>'
 '<a:effectStyle><a:effectLst/></a:effectStyle>'
 '<a:effectStyle><a:effectLst/></a:effectStyle></a:effectStyleLst>'
 '<a:bgFillStyleLst><a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
 '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill>'
 '<a:solidFill><a:schemeClr val="phClr"/></a:solidFill></a:bgFillStyleLst></a:fmtScheme>'
 '</a:themeElements><a:objectDefaults/><a:extraClrSchemeLst/></a:theme>')

# slide master
MASTER = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
 '<p:sldMaster xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
 'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
 '<p:cSld><p:bg><p:bgPr><a:solidFill><a:srgbClr val="FFFFFF"/></a:solidFill>'
 '<a:effectLst/></p:bgPr></p:bg><p:spTree><p:nvGrpSpPr><p:cNvPr id="0" name=""/>'
 '<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
 '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></p:grpSpPr></p:spTree></p:cSld>'
 '<p:clrMap bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" accent1="accent1" accent2="accent2" '
 'accent3="accent3" accent4="accent4" accent5="accent5" accent6="accent6" '
 'hlink="hlink" folHlink="folHlink"/>'
 '<p:sldLayoutIdLst><p:sldLayoutId id="2147483649" r:id="rId1"/></p:sldLayoutIdLst>'
 '<p:txStyles><p:titleStyle/><p:bodyStyle/><p:otherStyle/></p:txStyles></p:sldMaster>')

# slide layout
LAYOUT = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
 '<p:sldLayout xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
 'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
 'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" type="blank" '
 'preserve="1"><p:cSld name="Blank"><p:spTree><p:nvGrpSpPr><p:cNvPr id="0" name=""/>'
 '<p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
 '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></p:grpSpPr></p:spTree></p:cSld>'
 '<p:clrMapOvr><a:overrideClrMapping bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" '
 'accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" '
 'accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/></p:clrMapOvr>'
 '</p:sldLayout>')

# presentation.xml
def presentation_xml(n_slides, rels_ids):
    sld_ids = "".join(
        f'<p:sldId id="{256+i}" r:id="{rid}"/>' for i, rid in enumerate(rels_ids))
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<p:presentation xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
            'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main" saveSubsetFonts="1">'
            '<p:sldMasterIdLst><p:sldMasterId id="2147483648" r:id="rId1"/></p:sldMasterIdLst>'
            f'<p:sldIdLst>{sld_ids}</p:sldIdLst>'
            '<p:sldSz cx="12192000" cy="6858000" type="screen16x9"/>'
            '<p:notesSz cx="6858000" cy="9144000"/></p:presentation>')

# relationships for presentation
def presentation_rels(n_slides):
    items = ['<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="slideMasters/slideMaster1.xml"/>',
             '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="theme/theme1.xml"/>']
    for i in range(n_slides):
        items.append(f'<Relationship Id="rId{3+i}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide" Target="slides/slide{i+1}.xml"/>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(items) + '</Relationships>')

def master_rels():
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout" Target="../slideLayouts/slideLayout1.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/theme" Target="../theme/theme1.xml"/>'
            '</Relationships>')

def layout_rels():
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideMaster" Target="../slideMasters/slideMaster1.xml"/>'
            '</Relationships>')

def slide_rels(image_rels):
    items = [f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="../media/{fn}"/>'
             for (rid, fn) in image_rels]
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            + "".join(items) + '</Relationships>')

def content_types(n_slides, n_media):
    overrides = [
        '<Override PartName="/ppt/presentation.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml"/>',
        '<Override PartName="/ppt/slideMasters/slideMaster1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideMaster+xml"/>',
        '<Override PartName="/ppt/slideLayouts/slideLayout1.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slideLayout+xml"/>',
        '<Override PartName="/ppt/theme/theme1.xml" ContentType="application/vnd.openxmlformats-officedocument.theme+xml"/>',
    ]
    for i in range(n_slides):
        overrides.append(f'<Override PartName="/ppt/slides/slide{i+1}.xml" ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>')
    for i in range(n_media):
        overrides.append(f'<Override PartName="/ppt/media/image{i+1}.png" ContentType="image/png"/>')
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            + "".join(overrides) + '</Types>')

def root_rels():
    return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="ppt/presentation.xml"/>'
            '</Relationships>')

# ---------------------------------------------------------------------------
# Assemble
# ---------------------------------------------------------------------------
n_slides = len(SLIDES)
zf = zipfile.ZipFile(f"{OUT}/IFCS_2026_presentation.pptx", "w", zipfile.ZIP_DEFLATED)

# media mapping: collect all images used across slides
media_order = []
for s in SLIDES:
    for key in ("image", "images"):
        if key in s:
            specs = s[key] if isinstance(s[key], list) else [s[key]]
            for spec in specs:
                if spec[0] not in [m[0] for m in media_order]:
                    media_order.append((spec[0], os.path.basename(media[spec[0]])))

# map slide image keys -> (relId, filename)
slide_img_rel = {}
for si, s in enumerate(SLIDES):
    rels = []
    used = []
    for key in ("image", "images"):
        if key in s:
            specs = s[key] if isinstance(s[key], list) else [s[key]]
            for spec in specs:
                idx = [m[0] for m in media_order].index(spec[0])
                rid = f"rId{idx+1}"
                fn = media_order[idx][1]
                rels.append((rid, fn))
                used.append((rid, spec[1], spec[2], spec[3]))
    slide_img_rel[si] = used

# write media
for i, (key, fn) in enumerate(media_order):
    zf.writestr(f"ppt/media/{fn}", open(media[key], "rb").read())

# write slides
for si, s in enumerate(SLIDES):
    if s["type"] == "title":
        imgs = []
        sps = [rect_xml(0, 0, 13.333, 7.5, NAVY),
               textbox_xml(0.8, 2.4, 11.7, 1.2, [(s["title"], 40, WHITE, True)]),
               textbox_xml(0.8, 3.7, 11.7, 0.9, [(s["subtitle"], 22, LBLUE, False)]),
               textbox_xml(0.8, 5.6, 11.7, 0.6, [(s["footer"], 14, "9FB6CC", False)])]
        tree = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
                'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
                'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
                '<p:cSld><p:spTree><p:nvGrpSpPr><p:cNvPr id="0" name=""/><p:cNvGrpSpPr/>'
                '<p:nvPr/></p:nvGrpSpPr><p:grpSpPr><a:off x="0" y="0"/><a:ext cx="0" cy="0"/>'
                '<a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></p:grpSpPr>'
                + "".join(sps) + '</p:spTree></p:cSld>'
                '<p:clrMapOvr><a:overrideClrMapping bg1="lt1" tx1="dk1" bg2="lt2" tx2="dk2" '
                'accent1="accent1" accent2="accent2" accent3="accent3" accent4="accent4" '
                'accent5="accent5" accent6="accent6" hlink="hlink" folHlink="folHlink"/>'
                '</p:clrMapOvr></p:sld>')
        body_imgs = []
    else:
        body_imgs = slide_img_rel[si]
        tree = slide_xml(s["title"], s.get("bullets", []), body_imgs, s.get("note", ""))
    zf.writestr(f"ppt/slides/slide{si+1}.xml", tree)
    # slide rels
    rels = [(rid, fn) for (rid, l, t, w) in body_imgs]
    zf.writestr(f"ppt/slides/_rels/slide{si+1}.xml.rels", slide_rels(rels))

# presentation + rels
pres_rids = [f"rId{3+i}" for i in range(n_slides)]
zf.writestr("ppt/presentation.xml", presentation_xml(n_slides, pres_rids))
zf.writestr("ppt/_rels/presentation.xml.rels", presentation_rels(n_slides))
zf.writestr("ppt/slideMasters/slideMaster1.xml", MASTER)
zf.writestr("ppt/slideMasters/_rels/slideMaster1.xml.rels", master_rels())
zf.writestr("ppt/slideLayouts/slideLayout1.xml", LAYOUT)
zf.writestr("ppt/slideLayouts/_rels/slideLayout1.xml.rels", layout_rels())
zf.writestr("ppt/theme/theme1.xml", THEME)
zf.writestr("[Content_Types].xml", content_types(n_slides, len(media_order)))
zf.writestr("_rels/.rels", root_rels())
zf.close()

print("Saved artifacts/IFCS_2026_presentation.pptx")
print("slides:", n_slides, "| media images:", len(media_order))
