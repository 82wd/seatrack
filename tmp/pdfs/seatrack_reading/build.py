from pathlib import Path
import re, html
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, KeepTogether, Table, TableStyle
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.colors import HexColor, white
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader

ROOT=Path('/Users/lzy/Desktop/GitHub/seatrack')
OUT=ROOT/'output/pdf'
for name,file in [('CJK','Arial Unicode.ttf'),('Body','Arial.ttf'),('Bold','Arial Bold.ttf'),('Mono','Courier New.ttf')]:
    pdfmetrics.registerFont(TTFont(name,'/System/Library/Fonts/Supplemental/'+file))
navy=HexColor('#18344B');teal=HexColor('#007A80');ink=HexColor('#202B36');muted=HexColor('#566573')
W,H=A4;M=48;CW=W-2*M

def clean(t):return t.replace('–','-').replace('—','-').replace('\u2011','-')
def esc(t):return html.escape(clean(t))
def ismain(s):return bool(re.match(r'^[一二三四五六]、',s) or re.match(r'^[1-6]\. ',s))
def issub(s):return bool(re.match(r'^\d\.\d\s',s))

def code_markup(s):
    result=[];last=None;chunk=''
    for c in clean(s):
        font='Mono' if ord(c)<128 else 'CJK'
        if font!=last and chunk:
            result.append(f'<font name="{last}">{html.escape(chunk).replace(" ","&#160;")}</font>');chunk=''
        last=font;chunk+=c
    if chunk:result.append(f'<font name="{last}">{html.escape(chunk).replace(" ","&#160;")}</font>')
    return ''.join(result)

def line_width(s,size):return sum(pdfmetrics.stringWidth(c,'Mono' if ord(c)<128 else 'CJK',size) for c in clean(s))

def formula(lines):
    lines=[x[4:] if x.startswith('    ') else x for x in lines]
    maxw=max(line_width(x,11) for x in lines)
    size=min(11,11*(CW-28)/max(maxw,1))
    size=max(size,9)
    st=ParagraphStyle('formula',fontName='Mono',fontSize=size,leading=size*1.65,textColor=ink,spaceAfter=0,splitLongWords=True)
    ps=[Paragraph(code_markup(x),st) for x in lines]
    t=Table([[ps]],colWidths=[CW],hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),HexColor('#F1F5F8')),('BOX',(0,0),(-1,-1),.5,HexColor('#D8E3EB')),('LEFTPADDING',(0,0),(-1,-1),12),('RIGHTPADDING',(0,0),(-1,-1),12),('TOPPADDING',(0,0),(-1,-1),10),('BOTTOMPADDING',(0,0),(-1,-1),10)]))
    return [t,Spacer(1,11)]

class Doc(SimpleDocTemplate):
    def afterFlowable(self,flowable):
        if hasattr(flowable,'bookmark'):
            self.canv.bookmarkPage(flowable.bookmark)
            self.canv.addOutlineEntry(flowable.heading,flowable.bookmark,level=flowable.level,closed=False)

def build(lang,name,outname):
    cn=lang=='cn'; font='CJK' if cn else 'Body'; bold='CJK' if cn else 'Bold'
    txt=Path('/Users/lzy/Desktop',name).read_text()
    lines=txt.splitlines(); mainheads=[x for x in lines if ismain(x)]
    style=ParagraphStyle('body',fontName=font,fontSize=12.5 if cn else 12,leading=21 if cn else 19,spaceAfter=11,textColor=ink,wordWrap='CJK' if cn else None,splitLongWords=True)
    h1=ParagraphStyle('h1',fontName=bold,fontSize=21,leading=30,textColor=navy,spaceAfter=19,keepWithNext=True)
    h2=ParagraphStyle('h2',fontName=bold,fontSize=15,leading=23,textColor=teal,spaceBefore=14,spaceAfter=10,keepWithNext=True)
    small=ParagraphStyle('small',parent=style,fontSize=10.5,leading=17,textColor=muted)
    cover=ParagraphStyle('cover',fontName=bold,fontSize=30,leading=40,textColor=navy,spaceAfter=14)
    toc=ParagraphStyle('toc',parent=style,fontSize=13,leading=22,spaceAfter=12,textColor=teal)
    story=[Spacer(1,17),Paragraph('SEATrack',cover),Paragraph('核心模块计算详解' if cn else 'Core Module Computations',ParagraphStyle('subtitle',parent=cover,fontSize=23,leading=31)),Paragraph('AMG-LoRA · HMoE · '+('预测头' if cn else 'Prediction Head'),h2),Paragraph('中文版  /  2026-10-19' if cn else 'English edition  /  2026-10-19',small),Spacer(1,16)]
    first=next(i for i,l in enumerate(lines) if ismain(l))
    intro=[l for l in lines[:first] if len(l)>160][0]
    story += [Paragraph('阅读说明' if cn else 'Reading notes',h2),Paragraph(esc(intro),style),Spacer(1,4),Paragraph('目录' if cn else 'Contents',h2)]
    for n,t in enumerate(mainheads,1):story.append(Paragraph(f'<link href="#sec{n}" color="#007A80">{esc(t)}</link>',toc))
    count=0;lastsub=0;i=first
    while i<len(lines):
        line=lines[i];s=line.strip()
        if not s:i+=1;continue
        if ismain(line):
            count+=1;story.append(PageBreak())
            p=Paragraph(esc(line),h1);p.bookmark=f'sec{count}';p.heading=clean(line);p.level=0
            story.append(p);i+=1;continue
        if issub(line):
            lastsub+=1;p=Paragraph(esc(line),h2);p.bookmark=f'sub{lastsub}';p.heading=clean(line);p.level=1
            story.append(p);i+=1;continue
        if line.startswith('    '):
            block=[]
            while i<len(lines) and lines[i].startswith('    '):block.append(lines[i]);i+=1
            story.extend(formula(block));continue
        # Short labels introduce the next explanatory or formula block.
        st=small if s.startswith('/Users/') else style
        if len(s)<95 and (s.endswith((':','：')) or s in ['说明：','Note:']):st=h2
        story.append(Paragraph(esc(s),st));i+=1
    def page(canv,doc):
        canv.saveState();canv.setStrokeColor(HexColor('#D6E0E7'));canv.setLineWidth(.6)
        canv.line(M,H-36,W-M,H-36)
        canv.setFont(font,9);canv.setFillColor(muted)
        canv.drawString(M,H-26,'SEATrack  /  '+('核心模块解释' if cn else 'Core Module Explanations'))
        canv.line(M,38,W-M,38)
        canv.drawString(M,24,'AMG-LoRA  /  HMoE  /  '+('预测头' if cn else 'Prediction Head'))
        canv.drawRightString(W-M,24,('第 '+str(doc.page)+' 页') if cn else ('Page '+str(doc.page)))
        canv.restoreState()
    pdf=OUT/outname
    doc=Doc(str(pdf),pagesize=A4,rightMargin=M,leftMargin=M,topMargin=54,bottomMargin=53,title='SEATrack - '+('AMG-LoRA、HMoE 与预测头详解' if cn else 'AMG-LoRA, HMoE and Prediction Head'),author='SEATrack Study Notes',allowSplitting=True)
    doc.build(story,onFirstPage=page,onLaterPages=page)
    reader=PdfReader(pdf);extracted='\n'.join(p.extract_text() for p in reader.pages)
    for v in ['0.5875','1.0533','0.578125','332,312,128,256']:assert v in extracted,(outname,v)
    print(outname,len(reader.pages),'pages',pdf.stat().st_size,'bytes')
    (ROOT/'tmp/pdfs/seatrack_reading'/f'{lang}-extracted.txt').write_text(extracted)

build('cn','SEATrack_AMG-LoRA_HMoE_预测头_中文版.txt','SEATrack_AMG-LoRA_HMoE_预测头_中文版.pdf')
build('en','SEATrack_AMG-LoRA_HMoE_PredictionHead_English.txt','SEATrack_AMG-LoRA_HMoE_PredictionHead_English.pdf')
