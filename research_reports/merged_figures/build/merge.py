import zipfile, posixpath, copy, json
from pathlib import Path
import xml.etree.ElementTree as E

ROOT=Path(__file__).resolve().parents[1]
FILES=[Path('C:/Users/梁嘎嘎/Desktop/BRIGHT_model_formulas_v11_formula_final.pptx'),Path('D:/software/dowmload/BRIGHT-master/research_reports/figures/bright_data_map/BRIGHT_data_geographic_distribution_no_country_borders_editable.pptx'),Path('C:/Users/梁嘎嘎/Desktop/BRIGHT_Test_18cases_overview.pptx'),Path('C:/Users/梁嘎嘎/Desktop/BRIGHT_Test_A1_error_analysis_zoom.pptx')]
P='http://schemas.openxmlformats.org/presentationml/2006/main'; A='http://schemas.openxmlformats.org/drawingml/2006/main'; R='http://schemas.openxmlformats.org/officeDocument/2006/relationships'; REL='http://schemas.openxmlformats.org/package/2006/relationships'; CT='http://schemas.openxmlformats.org/package/2006/content-types'
for pre,ns in [('p',P),('a',A),('r',R)]: E.register_namespace(pre,ns)
def xml(x):
 if x.tag.startswith('{'+REL+'}'):E.register_namespace('',REL)
 elif x.tag.startswith('{'+CT+'}'):E.register_namespace('',CT)
 return E.tostring(x,encoding='utf-8',xml_declaration=True)
out={}; types=E.Element('{'+CT+'}Types'); defaults={}; overrides={}; slides=[]; masters=[]; report=[]
W,H=15240000,10287000
for di,f in enumerate(FILES):
 z=zipfile.ZipFile(f); names=set(z.namelist()); prefix=f'ppt/import{di}/'
 def mapped(n):return prefix+n.lstrip('/')
 pres=E.fromstring(z.read('ppt/presentation.xml')); sz=pres.find('{'+P+'}sldSz'); w,h=int(sz.get('cx')),int(sz.get('cy'))
 rr=E.fromstring(z.read('ppt/_rels/presentation.xml.rels')); links={r.get('Id'):posixpath.normpath(posixpath.join('ppt',r.get('Target'))) for r in rr}
 for item in pres.find('{'+P+'}sldIdLst'):slides.append(mapped(links[item.get('{'+R+'}id')]))
 for item in pres.find('{'+P+'}sldMasterIdLst'):masters.append(mapped(links[item.get('{'+R+'}id')]))
 for el in E.fromstring(z.read('[Content_Types].xml')):
  if el.tag.endswith('Default'): defaults[el.get('Extension')]=el.get('ContentType')
  else: overrides['/'+mapped(el.get('PartName').lstrip('/'))]=el.get('ContentType')
 for n in names:
  if n.endswith('/') or n in ('[Content_Types].xml','_rels/.rels'):continue
  data=z.read(n)
  if n.endswith('.rels'):
   tree=E.fromstring(data)
   base='' if n=='_rels/.rels' else posixpath.dirname(posixpath.dirname(n))
   for rel in tree:
    if rel.get('TargetMode')=='External':continue
    target=rel.get('Target'); resolved=target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join(base,target))
    rel.set('Target','/'+mapped(resolved))
   data=xml(tree)
  if n.startswith('ppt/slides/slide') and n.endswith('.xml'):
   tree=E.fromstring(data); st=tree.find('.//{'+P+'}spTree'); kids=list(st)[2:]
   if kids:
    group=E.Element('{'+P+'}grpSp'); nv=E.SubElement(group,'{'+P+'}nvGrpSpPr');E.SubElement(nv,'{'+P+'}cNvPr',id='2147483000',name='Original editable content');E.SubElement(nv,'{'+P+'}cNvGrpSpPr');E.SubElement(nv,'{'+P+'}nvPr')
    pr=E.SubElement(group,'{'+P+'}grpSpPr'); xf=E.SubElement(pr,'{'+A+'}xfrm');scale=min(W/w,H/h);nw,nh=round(w*scale),round(h*scale)
    for tag,attrs in [('off',{'x':str((W-nw)//2),'y':str((H-nh)//2)}),('ext',{'cx':str(nw),'cy':str(nh)}),('chOff',{'x':'0','y':'0'}),('chExt',{'cx':str(w),'cy':str(h)})]:E.SubElement(xf,'{'+A+'}'+tag,attrs)
    for kid in kids:st.remove(kid);group.append(kid)
    st.append(group)
   # Bake uniform scale into native geometry and type sizes to avoid renderer-dependent group text scaling.
   if kids:
    st.remove(group)
    for kid in kids:
     for el in kid.iter():
      for attr in ('x','y','cx','cy','sz','marL','marR','marT','marB','lIns','rIns','tIns','bIns'):
       if attr in el.attrib:
        try:el.set(attr,str(round(int(el.get(attr))*scale)))
        except ValueError:pass
     xf=kid.find('./{'+P+'}spPr/{'+A+'}xfrm')
     if xf is None:xf=kid.find('./{'+P+'}grpSpPr/{'+A+'}xfrm')
     if xf is None:xf=kid.find('./{'+P+'}xfrm')
     if xf is not None:
      off=xf.find('{'+A+'}off')
      if off is not None:off.set('x',str(int(off.get('x'))+(W-nw)//2));off.set('y',str(int(off.get('y'))+(H-nh)//2))
     st.append(kid)
   data=xml(tree)
  out[mapped(n)]=data
 report.append({'file':str(f),'slides':len(pres.find('{'+P+'}sldIdLst')),'original_size':[w,h]})
pres=E.Element('{'+P+'}presentation');ml=E.SubElement(pres,'{'+P+'}sldMasterIdLst');sl=E.SubElement(pres,'{'+P+'}sldIdLst');rels=E.Element('{'+REL+'}Relationships')
for i,m in enumerate(masters):
 rid='m'+str(i);E.SubElement(ml,'{'+P+'}sldMasterId',{'id':str(2147483648+i),'{'+R+'}id':rid});E.SubElement(rels,'{'+REL+'}Relationship',Id=rid,Type=R+'/slideMaster',Target='/'+m)
for i,s in enumerate(slides):
 rid='s'+str(i);E.SubElement(sl,'{'+P+'}sldId',{'id':str(256+i),'{'+R+'}id':rid});E.SubElement(rels,'{'+REL+'}Relationship',Id=rid,Type=R+'/slide',Target='/'+s)
E.SubElement(pres,'{'+P+'}sldSz',cx=str(W),cy=str(H));E.SubElement(pres,'{'+P+'}notesSz',cx='6858000',cy='9144000')
out['ppt/presentation.xml']=xml(pres);out['ppt/_rels/presentation.xml.rels']=xml(rels)
rr=E.Element('{'+REL+'}Relationships');E.SubElement(rr,'{'+REL+'}Relationship',Id='rId1',Type=R+'/officeDocument',Target='ppt/presentation.xml');out['_rels/.rels']=xml(rr)
overrides['/ppt/presentation.xml']='application/vnd.openxmlformats-officedocument.presentationml.presentation.main+xml'
for ext,ct in defaults.items():E.SubElement(types,'{'+CT+'}Default',Extension=ext,ContentType=ct)
for n,ct in overrides.items():E.SubElement(types,'{'+CT+'}Override',PartName=n,ContentType=ct)
out['[Content_Types].xml']=xml(types)
# Standard slide locations improve compatibility with third-party PPTX readers.
renames={s:f'ppt/slides/slide{i+1}.xml' for i,s in enumerate(slides)}
for old,new in list(renames.items()):
 oldrel=posixpath.dirname(old)+'/_rels/'+posixpath.basename(old)+'.rels'
 if oldrel in out:renames[oldrel]=posixpath.dirname(new)+'/_rels/'+posixpath.basename(new)+'.rels'
fixed={}
for name,data in out.items():
 if name.endswith('.rels'):
  root=E.fromstring(data)
  for r in root:
   t=r.get('Target','').lstrip('/')
   if t in renames:r.set('Target','/'+renames[t])
  data=xml(root)
 elif name=='[Content_Types].xml':
  root=E.fromstring(data)
  for el in root:
   n=el.get('PartName','').lstrip('/')
   if n in renames:el.set('PartName','/'+renames[n])
  data=xml(root)
 fixed[renames.get(name,name)]=data
out=fixed
with zipfile.ZipFile(ROOT/'build/candidate.pptx','w',zipfile.ZIP_DEFLATED) as z:
 for n,b in out.items():z.writestr(n,b)
(ROOT/'build/sources.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
print('merged',len(slides),'slides')
