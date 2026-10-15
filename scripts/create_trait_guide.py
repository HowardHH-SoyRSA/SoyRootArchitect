"""Build the illustrated trait reference from a completed layout-v2 bundle.

Run --plots with the program's Python; run --pdf with reportlab installed.
The two steps share only ordinary PNG/CSV/JSON files, not Python environments.
"""
from pathlib import Path
import argparse
import csv
import json
import math
from xml.sax.saxutils import escape


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def fmt(value, digits=3):
    v = number(value)
    if not math.isfinite(v):
        return "NA"
    return f"{v:,.{digits}f}".rstrip("0").rstrip(".") if digits else f"{v:,.0f}"


def make_plots(bundle, assets):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.spatial import ConvexHull
    assets.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":10, "axes.spines.top":False,
                         "axes.spines.right":False, "savefig.dpi":220})
    summary = pd.read_csv(bundle/'tables/architecture_summary.csv')
    roots = pd.read_csv(bundle/'tables/root_traits.csv')
    mapping = pd.read_csv(bundle/'tables/root_label_map.csv')
    metadata = json.loads((bundle/'support/metadata.json').read_text(encoding='utf-8'))
    with np.load(bundle/'support/analysis_geometry.npz', allow_pickle=False) as data:
        points=data['positions']; labels=data['root_labels']; orders=data['root_orders']
    eligible=mapping[mapping.root_id.isin(roots.root_id)].numeric_label.to_numpy(int)
    keep=np.isin(labels,eligible)
    colors={0:'#1355c5',1:'#c72ba5',2:'#008d6b',3:'#7951b0'}
    top=np.asarray(metadata['measurement_policy']['immutable_primary_top'])
    gravity=np.asarray(metadata.get('gravity_vector',[0,0,-1.]))
    gravity=gravity/np.linalg.norm(gravity)
    helper=np.eye(3)[np.argmin(np.abs(gravity))]; u=np.cross(gravity,helper);u/=np.linalg.norm(u);v=np.cross(gravity,u)
    xy=np.column_stack(((points-top)@u,(points-top)@v)); depth=(points-top)@gravity
    fig,axes=plt.subplots(1,2,figsize=(9,5.4),layout='constrained')
    for order in sorted(roots.root_order.unique()):
        ids=np.flatnonzero(keep&(orders==order)); ids=ids[::max(1,len(ids)//18000)]
        axes[0].scatter(xy[ids,0],depth[ids],s=.55,color=colors.get(order,'#bb8500'),rasterized=True,label=f'Order {order}')
        axes[1].scatter(xy[ids,0],xy[ids,1],s=.55,color=colors.get(order,'#bb8500'),rasterized=True)
    hull=ConvexHull(xy[keep]); ring=xy[keep][np.r_[hull.vertices,hull.vertices[0]]]
    axes[1].plot(ring[:,0],ring[:,1],color='#3b4858',lw=1,ls='--')
    axes[0].scatter([0],[0],marker='*',c='#242b35',s=70,zorder=4)
    axes[0].invert_yaxis();axes[0].set(xlabel='Horizontal position (mm)',ylabel='Depth below immutable top (mm)',title='Front projection')
    axes[1].set(xlabel='Horizontal position (mm)',ylabel='Horizontal position (mm)',title='Horizontal envelope')
    for ax in axes:ax.set_aspect('equal',adjustable='datalim')
    axes[0].legend(markerscale=5,frameon=False,loc='lower right')
    fig.savefig(assets/'extent.png',facecolor='white');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
    dist=pd.read_csv(bundle/'tables/diameter_distribution.csv')
    for order in sorted(roots.root_order.unique()):
        x=dist[dist.scope==f'order_{order}']; color=colors.get(order,'#bb8500')
        axes[0].step(x.mean_diameter,x.cumulative_fraction,where='post',label=f'Order {order}',color=color)
        vals=roots.loc[roots.root_order==order,'mean_diameter'].dropna()
        axes[1].scatter(np.full(len(vals),order),vals,s=13,alpha=.65,color=color)
        if len(vals):
            q=np.quantile(vals,[.1,.25,.5,.75,.9]);axes[1].plot([order,order],q[[0,4]],color='#222',lw=1)
            axes[1].plot([order,order],q[[1,3]],color='#222',lw=5);axes[1].scatter([order],[q[2]],color='white',s=13,zorder=5)
    axes[0].set(xlabel='Per-root mean diameter (mm)',ylabel='Cumulative fraction',ylim=(0,1.03));axes[0].legend(frameon=False)
    axes[1].set(xlabel='Root order',ylabel='Per-root mean diameter (mm)');axes[1].set_xticks(sorted(roots.root_order.unique()))
    fig.savefig(assets/'diameter.png',facecolor='white');plt.close(fig)
    by_order=summary[summary.scope.str.startswith('order_')]
    fig,axes=plt.subplots(1,2,figsize=(9,3.3),layout='constrained')
    x=np.arange(len(by_order)); labels2=[s.replace('order_','O') for s in by_order.scope]
    axes[0].bar(x,by_order.root_count,color='#008d6b',label='Eligible')
    axes[0].bar(x,by_order.excluded_root_count,bottom=by_order.root_count,color='#cbd1d8',label='Excluded')
    axes[0].set(ylabel='Roots',xticks=x,xticklabels=labels2);axes[0].legend(frameon=False)
    axes[1].bar(x-.18,by_order.volume_frustum,width=.36,color='#008d6b',label='Frustum')
    axes[1].bar(x+.18,by_order.mesh_volume,width=.36,color='#7951b0',label='Mesh integral')
    axes[1].set(ylabel='Volume (mm³)',xticks=x,xticklabels=labels2);axes[1].legend(frameon=False)
    fig.savefig(assets/'coverage_volume.png',facecolor='white');plt.close(fig)
    gaps=pd.read_csv(bundle/'tables/branch_spacing.csv'); gaps=gaps[gaps.scope=='laterals']
    fig,axes=plt.subplots(1,2,figsize=(9,3.2),layout='constrained')
    axes[0].hist(gaps.distance,bins=18,color='#008d6b',edgecolor='white');axes[0].set(xlabel='Within-parent spacing (mm)',ylabel='Intervals')
    for order in sorted(roots.root_order.unique()):
        vals=roots.loc[roots.root_order==order,'tip_gravity_angle_deg'].dropna()
        axes[1].scatter(np.full(len(vals),order),vals,s=12,alpha=.6,color=colors.get(order,'#bb8500'))
    axes[1].set(xlabel='Root order',ylabel='Tip-gravity angle (degrees)',ylim=(0,180),xticks=sorted(roots.root_order.unique()))
    fig.savefig(assets/'spacing_angles.png',facecolor='white');plt.close(fig)


def make_pdf(bundle, assets, destination):
    from reportlab.pdfgen import canvas
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak, KeepTogether
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_LEFT
    from reportlab.graphics.shapes import Drawing, Line, String, Circle, Polygon, PolyLine
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    font=Path('C:/Windows/Fonts/arial.ttf'); bold=Path('C:/Windows/Fonts/arialbd.ttf')
    pdfmetrics.registerFont(TTFont('Guide',str(font)));pdfmetrics.registerFont(TTFont('GuideBold',str(bold)))
    pdfmetrics.registerFontFamily('Guide',normal='Guide',bold='GuideBold')
    styles=getSampleStyleSheet()
    for name in ['Normal','BodyText']:
        styles[name].fontName='Guide';styles[name].fontSize=9.2;styles[name].leading=13
    styles.add(ParagraphStyle('TitleG',fontName='GuideBold',fontSize=27,leading=31,textColor=colors.HexColor('#17293a'),spaceAfter=15))
    styles.add(ParagraphStyle('HeadG',fontName='GuideBold',fontSize=18,leading=22,textColor=colors.HexColor('#17293a'),spaceAfter=12))
    styles.add(ParagraphStyle('SubG',fontName='GuideBold',fontSize=11,leading=15,textColor=colors.HexColor('#00846d'),spaceBefore=10,spaceAfter=5))
    styles.add(ParagraphStyle('SmallG',fontName='Guide',fontSize=7.5,leading=10,spaceAfter=4))
    styles.add(ParagraphStyle('CellG',fontName='Guide',fontSize=8,leading=10.5,wordWrap='CJK'))
    styles.add(ParagraphStyle('CaptionG',fontName='Guide',fontSize=8,leading=11,textColor=colors.HexColor('#516071'),spaceAfter=10))
    tables=bundle/'tables';support=bundle/'support'
    summary=read_csv(tables/'architecture_summary.csv');scopes={r['scope']:r for r in summary};system=scopes['system']; lateral=scopes['laterals']
    roots=read_csv(tables/'root_traits.csv');inventory=read_csv(tables/'root_inventory.csv')
    meta=json.loads((support/'metadata.json').read_text(encoding='utf-8'))
    input_name=Path(meta.get('input_path') or meta.get('config',{}).get('input_path') or bundle.name).name
    catalog=read_csv(tables/'trait_dictionary.csv')
    story=[]; w=499
    def para(text,style='BodyText'):
        story.append(Paragraph(text,styles[style]));story.append(Spacer(1,6))
    def heading(title,subtitle=None):
        story.append(Paragraph(title,styles['HeadG']))
        if subtitle:para(subtitle)
    def table(rows,widths=None,compact=False):
        cells=[[Paragraph(escape(str(v)),styles['CellG']) for v in row] for row in rows]
        t=Table(cells,colWidths=widths or [w/len(rows[0])]*len(rows[0]),repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e8f1f1')),('VALIGN',(0,0),(-1,-1),'TOP'),
                              ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#00846d')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f7f9')]),
                              ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5 if compact else 6),('BOTTOMPADDING',(0,0),(-1,-1),5 if compact else 6)]))
        story.append(t);story.append(Spacer(1,10))
    def picture(name,height):
        from PIL import Image as PILImage
        path=assets/name
        with PILImage.open(path) as im:pw,ph=im.size
        width=min(w,height*pw/ph);story.append(Image(str(path),width=width,height=width*ph/pw));story.append(Spacer(1,8))
    def page():story.append(PageBreak())
    def example_table(fields,selected=('system','laterals','order_0','order_1','order_2','order_3')):
        rows=[['Scope',*[label for key,label in fields]]]
        for scope in selected:
            if scope in scopes:
                rows.append([scope,*[fmt(scopes[scope].get(key)) for key,label in fields]])
        table(rows,[91]+[(w-91)/len(fields)]*len(fields))
    def schematic(kind):
        d=Drawing(w,160);dark=colors.HexColor('#17293a');green=colors.HexColor('#008d6b');pink=colors.HexColor('#c72ba5')
        def line(x1,y1,x2,y2,c=dark,lw=2):d.add(Line(x1,y1,x2,y2,strokeColor=c,strokeWidth=lw))
        def text(x,y,s,size=9):d.add(String(x,y,s,fontName='Guide',fontSize=size,fillColor=dark))
        if kind=='form':
            coords=[30,120,70,105,112,98,150,68,194,59,230,30]
            d.add(PolyLine(coords,strokeColor=green,strokeWidth=5));line(30,120,230,30,pink,1)
            for i in range(0,len(coords),2):d.add(Circle(coords[i],coords[i+1],3,fillColor=green,strokeColor=green))
            text(40,140,'Arc length L follows the centerline');text(40,12,'Chord C joins start to tip; tortuosity = L/C')
            d.add(Polygon([285,55,285,112,435,96,435,71],strokeColor=green,fillColor=colors.HexColor('#e5f3ed')))
            line(285,83,435,83,dark,1);line(285,83,285,112,pink,2);line(435,83,435,96,pink,2)
            text(292,119,'r1');text(438,98,'r2');text(345,68,'h');text(285,35,'V = pi h (r1² + r1 r2 + r2²) / 3')
        elif kind=='branch':
            line(55,140,55,20,colors.HexColor('#1355c5'),5)
            for y,end in [(115,165),(80,150),(45,135)]:line(55,y,end,y-20,pink,3)
            for x in [105,140]:line(x,105-(x-55)*20/110,x+15,140,green,3)
            text(195,125,'Count children on every recorded parent.');text(195,100,'Density = child count / distinct parent length')
            text(195,75,'Spacing = consecutive insertion arc gaps');text(195,50,'Never join intervals across different parents.');text(195,25,'Coincident insertions stay as zero gaps.')
        elif kind=='angle':
            for x,title in [(65,'Tip vs gravity'),(225,'Chord vs gravity'),(385,'Base vs parent')]:
                line(x,125,x,25,dark,2);line(x,125,x+48,55,green,3);text(x-30,6,title)
                d.add(Circle(x,125,3,fillColor=pink,strokeColor=pink))
            text(22,145,'Directed vectors span 0-180 degrees; base-parent axial angle spans 0-90 degrees.')
        story.append(d);story.append(Spacer(1,8))
    para('SOYROOTARCHITECT  /  TRAIT REFERENCE', 'SubG')
    story.append(Paragraph('Root architecture,<br/>measured across orders',styles['TitleG']))
    para('Definitions, export structure and a worked example', 'SubG')
    para(escape(input_name)+' | Measurements and QC from the selected export bundle')
    picture('extent.png',300)
    table([['Eligible roots','Excluded topology roots','Measured length (mm)','Detected eligible orders'],
           [fmt(system['root_count'],0),fmt(system['excluded_root_count'],0),fmt(system['length_sum']),', '.join(r['scope'].replace('order_','') for r in summary if r['scope'].startswith('order_') and number(r['root_count'])>0)]])
    para('The plots show the eligible assigned subset. The complete hierarchy, source geometry and excluded identities remain available for review. This is an example of the measurement contract, not a claim that every segmented branch is biologically correct.','CaptionG')
    page();heading('1. Scope, eligibility and units')
    para('<b>Eligibility:</b> measured root length must be finite and strictly greater than zero. Noise-only topology placeholders are excluded. Positive partial measurements remain eligible, with their completeness flags visible.')
    table([['Scope','Included roots','Geometry used'],['system','Primary plus all detected lateral orders','Vertices assigned to eligible roots'],['laterals','All orders greater than zero','Only their assigned vertices'],['order_N','One recorded order, including order_0','Only vertices assigned to that order']],[75,185,239])
    para('All coordinates and linear measurements in this example are already calibrated to mm. Area is mm², volume is mm³, branching density is roots/mm, and envelope length density is mm/mm³. Multiply roots/mm by 10 to express roots/cm; multiply mm/mm³ by 100 to express cm/cm³. Do not apply the STL calibration twice.')
    para('Excluded geometry includes unassigned, uncertain and nodule vertices as well as roots with missing/zero length. A mesh triangle enters a scope only if all three corners pass that scope. No triangle is invented, capped or repaired.')
    picture('coverage_volume.png',205)
    para('Left: excluded identities remain visible in QC. Right: independent volume methods can disagree because their support and geometric assumptions differ. Mesh integrals of separate root patches are not required to add to the union integral.','CaptionG')
    example_table([('root_count','Eligible'),('excluded_root_count','Excluded'),('length_sum','Length (mm)')])
    page();heading('2. Root form: length, diameter and tortuosity')
    schematic('form')
    para('<b>Length</b> is the path arc length; <b>chord length</b> is straight-line start-to-tip distance; <b>tortuosity</b> is their ratio. Supported parent connectors can contribute to root length. Exposed-body length, connector length and preserved topology length are exported separately.')
    para('<b>Per-root diameter</b> is twice the radius profile from final assigned exposed-body support. Mean and median use arc-length weighting where supported; minimum and maximum are finite station extrema. A station-statistics fallback is named explicitly. Frustum volume uses exposed-body radii, excluding internal parent connectors.')
    example_table([('mean_length','Mean length (mm)'),('mean_chord_length','Mean chord (mm)'),('mean_tortuosity','Mean tortuosity'),('diameter_mean','Mean diameter (mm)')])
    para('Within a scope, form means give each root one vote. Missing values are omitted from that particular mean, and availability counts remain visible. No missing radius, length or volume is imputed. An entirely unmeasurable scope retains missing totals.')
    para('Stored topology paths can outlive measurable centerlines. They remain in the hierarchy and centerlines table for editing; retained unsupported paths must not be interpreted as measured length.','CaptionG')
    page();heading('3. Branching density and insertion spacing')
    schematic('branch')
    para('Insertion positions are projected onto the closest segment of the current recorded parent polyline. Arc distance is measured from that parent\'s start. Parent IDs are respected even when manual root-order labels do not form the usual generation sequence.')
    para('<b>Branching density</b> = eligible child count / sum of distinct eligible parent lengths. Each parent contributes once. If any required parent is unmeasurable or unavailable, density is missing and the number of affected children is reported.')
    para('<b>Mean inter-branch distance</b> pools consecutive gaps within parents. Parents with one child supply no interval. Terminal margins are excluded. <b>Spacing CV</b> is sample SD / mean; it requires at least two gaps and a positive mean.')
    example_table([('lateral_density','Roots/mm'),('mean_interbranch','Mean gap (mm)'),('interbranch_cv','Gap CV'),('coincident_gap_count','Zero/near-zero gaps')],('laterals','order_1','order_2','order_3'))
    para('The main measurement retains coincident sites. A distinct-site sensitivity merges positions within 1e-9 mm, then recomputes mean, CV and interval count. This is a geometry diagnostic, not an additional treatment-comparison test.')
    table([['All laterals','Main spacing','Distinct-site sensitivity'],['Mean gap (mm)',fmt(lateral['mean_interbranch']),fmt(lateral['distinct_site_mean_interbranch'])],['Interval count',fmt(lateral['interbranch_gap_count'],0),fmt(lateral['distinct_site_gap_count'],0)]])
    page();heading('4. Extent and compactness')
    picture('extent.png',285)
    para('<b>Depth</b> is the maximum downward projection from the immutable primary top. <b>Width</b> is the greatest pairwise horizontal distance, found on the 2D convex hull. <b>Depth/width</b> describes envelope shape, not tissue packing.')
    para('<b>Relative reach</b> is the P90 of per-root horizontal start-to-tip displacement divided by primary length. <b>Hull volume</b> encloses eligible assigned vertices and includes empty space. <b>Envelope length density</b> divides eligible total length by that hull volume.')
    example_table([('max_depth','Depth (mm)'),('max_width','Width (mm)'),('depth_width','Depth/width'),('hull_volume','Hull (mm³)')])
    para('Zero-dimensional, linear or planar point sets have a zero-volume hull and an explicit degenerate status. Ratios with zero/missing denominators remain missing. Curved roots may reach farther than their start-to-tip displacement.','CaptionG')
    page();heading('5. Diameter distribution across roots')
    picture('diameter.png',235)
    para('Each eligible root contributes its <b>per-root mean diameter</b> once. The empirical cumulative distribution is exported at every distinct value. It does not pool all longitudinal radius stations, which would overweight roots with more sampled stations.')
    para('Mean, median, P10, P90, interquartile range (P75 - P25), and CV are reported for each scope. Quantiles use linear interpolation, matching R type 7. CV uses sample SD and is undefined with fewer than two roots or a zero mean.')
    example_table([('diameter_mean','Mean (mm)'),('diameter_median','Median (mm)'),('diameter_p10','P10 (mm)'),('diameter_p90','P90 (mm)'),('diameter_cv','CV')])
    para('Plot: colored points are individual roots, thick black segments mark the IQR, white points mark the median, and thin lines span P10-P90. Root order 0 is the primary root; its across-root distribution contains one root and its CV is therefore missing.','CaptionG')
    page();heading('6. Directional angles and vector evidence')
    schematic('angle')
    para('Directions follow start-to-tip ordering. Tip-gravity uses the terminal tangent; chord-gravity uses the full start-to-tip vector. Tip-parent compares the child tip tangent with its local parent tangent. The primary root has gravity/chord angles but no parent-relative insertion angle.')
    para('The historical <b>tip_primary_angle_deg</b> name means child <b>base</b> tangent versus the local primary tangent. Its alias tip_angle_primary_deg has the same meaning. For higher-order roots this remains a primary reference; base_parent_angle_deg and tip_angle_parent_deg refer to the direct recorded parent.')
    para('The base-parent compatibility angle is axial (0-90 degrees). Directed gravity and reference-vector angles span 0-180 degrees. Requested/actual arc windows, vector starts, ends and components allow the measurements to be checked.')
    picture('spacing_angles.png',205)
    example_table([('mean_base_parent_angle_deg','Base-parent (deg)'),('mean_tip_gravity_angle_deg','Tip-gravity (deg)'),('mean_tip_start_gravity_angle_deg','Chord-gravity (deg)')])
    page();heading('7. Two volume methods and solidity')
    para('<b>Frustum volume</b> models supported exposed bodies as circular frusta. <b>Mesh volume</b> is the sum of absolute signed tetrahedral integrals by native connected component, using a component vertex as the integration origin. Both use only eligible segmented roots; raw whole-source volume is not a root trait.')
    example_table([('volume_frustum','Frustum (mm³)'),('mesh_volume','Mesh integral (mm³)'),('mesh_boundary_edges','Boundary edges'),('mesh_volume_origin_sensitivity','Origin range (mm³)')])
    para('A mesh integral is reliable as an enclosed-surface measurement only when the retained patch is nonempty, closed, consistently oriented and free of nonmanifold edges, degenerate faces and duplicate indexed faces. Root segmentation often opens patches. A numeric result is then marked <b>provisional</b>; it is never repaired into compliance.')
    para('Origin sensitivity is the range after shifting the integration origin by half the patch extent along each axis in both directions. It is a diagnostic, not a statistical confidence interval. A closed-volume integral is origin invariant.')
    example_table([('solidity_mesh','Mesh solidity'),('solidity_frustum','Frustum solidity'),('envelope_length_density','Length density (mm/mm³)'),('relative_horizontal_reach_p90','Relative reach P90')])
    para('Solidity divides the indicated volume by the convex-hull volume. Interpret it with the numerator\'s reliability and completeness. Neither estimate is clipped to [0,1], and a small envelope alone does not prove compact tissue packing.','CaptionG')
    page();heading('8. Measurement quality and biological QC')
    example_table([('root_count','Eligible'),('excluded_root_count','Excluded'),('length_estimate_complete_root_count','Complete length'),('radius_profile_complete_root_count','Complete radius'),('volume_estimate_complete_root_count','Complete frustum')])
    audit=meta.get('final_compliance_audit',{})
    para('The new measurements do not change ownership, hierarchy or source geometry. Collar exclusions, immutable-top origin checks, higher-order primary contact, disconnected child patches and child/parent length reviews retain their existing policies.')
    unresolved = audit.get('source_connectivity_status','') == 'unresolved_stl_connectivity' or audit.get('discrete_child_patch_status','') != 'clear'
    table([['Example export audit','Recorded value'],['Source connectivity',audit.get('source_connectivity_status','not reported')],['Compliance interpretation','Unresolved findings remain' if unresolved else 'Consult recorded native audits'],['Higher-order contact',audit.get('higher_order_primary_contact_status',meta.get('higher_order_primary_contact',{}).get('status','not reported'))],['Child surface patches',audit.get('discrete_child_patch_status',meta.get('final_surface_patch_audit',{}).get('status','not reported'))],['Segmented mesh volume status',system['mesh_volume_status']]], [205,294])
    flags={}
    for row in inventory:
        for flag in row.get('qc_flags','').split(';'):
            if flag:flags[flag]=flags.get(flag,0)+1
    table([['Most frequent root QC flags','Root count'],*[[k,str(v)] for k,v in sorted(flags.items(),key=lambda x:(-x[1],x[0]))[:9]]],[405,94])
    para('Unresolved findings remain unresolved. A child longer than its parent is review evidence and cannot, by itself, justify deletion or reparenting. Completeness counts refer to the eligible subset; excluded topology identities are counted separately.','CaptionG')
    page();heading('9. Reading and reproducing the export')
    table([['Location','Contents and role'],['tables/','Canonical CSV tables and one consolidated Excel workbook. Each table occurs once as a sheet; obsolete per-trait slices are merged.'],['support/','Hierarchy, RSML, metadata, original/indexed geometry, mappings, masks and provenance.'],['presentation_root_structure.ply','Canonical labelled display mesh with excluded disconnected noise physically removed.'],['primary_points.ply / lateral_points.ply','Retained full-resolution assigned point clouds with coordinates and colors; laterals include all orders. Written when the class contains points.'],['skeleton_original_overlay.ply','Side-by-side diagnostic; translated skeleton is not a source of measurement coordinates.'],['PNG files','Overview and angle views.']],[160,339])
    para('root_traits.csv contains only eligible measurements. root_inventory.csv retains every topology identity and its exclusion/QC status. centerlines.csv merges primary and lateral stations. architecture_summary.csv supplies all-order and per-order comparisons; branch tables and the diameter ECDF provide the underlying distributions.')
    para('The program\'s bundled editor supports both legacy and version-2 layouts. Other tools that expect root-level metadata or segmented_root_structure.ply need to adopt the new paths. Existing bundles are preserved.')
    para('<b>Source:</b> '+escape(str(meta.get('source',''))))
    para('<b>Immutable top (mm):</b> '+escape(str(meta.get('measurement_policy',{}).get('immutable_primary_top'))))
    reproduction = support/'reproduction.json'
    if not reproduction.exists():reproduction=bundle.parent/'reproduction.json'
    para('<b>Input SHA-256:</b> '+escape(json.loads(reproduction.read_text())['input_sha256']), 'SmallG')
    para('<b>Run time:</b> '+fmt(meta.get('stage_timings_seconds',{}).get('total'))+' seconds. Source hashes and frozen execution inputs are recorded with the example package.')
    para('This guide was generated from the exported CSV/JSON files. The following dictionary covers the exported fields; aliases and diagnostics are labeled. Group statistical tests, p-values and q-values require a separate experiment-level analysis and are not single-root traits.','CaptionG')
    page();heading('Field dictionary', 'Complete definitions for the exported measurement tables. Repeated fields share one definition; table membership is listed.')
    by_field={}
    for row in catalog:
        if row['field'] not in by_field:by_field[row['field']]={'definition':row['definition'],'tables':[]}
        by_field[row['field']]['tables'].append(row['table'])
    rows=[['Field','Definition']]
    for field,item in sorted(by_field.items()):
        rows.append([field,item['definition']+' Tables: '+', '.join(sorted(set(item['tables'])))+'.'])
    table(rows,[151,348],compact=True)
    destination.parent.mkdir(parents=True,exist_ok=True)
    def footer(c,doc):
        c.saveState();c.setFont('Guide',8);c.setFillColor(colors.HexColor('#647487'))
        c.drawString(48,27,'SoyRootArchitect | All-order traits | Worked example')
        c.drawRightString(A4[0]-48,27,str(doc.page));c.restoreState()
    doc=SimpleDocTemplate(str(destination),pagesize=A4,rightMargin=48,leftMargin=48,topMargin=43,bottomMargin=46,
                         title='SoyRootArchitect: measured traits across all root orders',author='SoyRootArchitect')
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(destination)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('bundle',type=Path);parser.add_argument('--assets',type=Path,required=True)
    parser.add_argument('--plots',action='store_true');parser.add_argument('--pdf',type=Path)
    args=parser.parse_args()
    if args.plots:make_plots(args.bundle,args.assets)
    if args.pdf:make_pdf(args.bundle,args.assets,args.pdf)
