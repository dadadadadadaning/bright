const fs = require('fs');
const path = require('path');
const deps = 'C:/Users/梁嘎嘎/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/';
const {chromium} = require(deps+'playwright');
const sharp = require(deps+'sharp');
const dir = __dirname;
(async()=>{
  const pagePath = path.resolve(dir,'../../tmp/pdfs/bright_figure_review/page-12.png');
  const meta = await sharp(pagePath).metadata();
  const data = fs.readFileSync(pagePath).toString('base64');
  let svg = fs.readFileSync(path.join(dir,'BRIGHT_visual_grammar_v1.svg'),'utf8');
  // PDF page coordinates refer to the verified 1298 x 1700 rendering.
  function tile(x,y,cx){
    return `<svg x="${x+12}" y="${y}" width="96" height="96" viewBox="${cx} 806 86 87" preserveAspectRatio="none" overflow="hidden"><image width="${meta.width}" height="${meta.height}" href="data:image/png;base64,${data}"/></svg>`;
  }
  svg = svg.replace(/<g transform="translate\(32,83\) scale\(1\)">[\s\S]*?<\/g>/, tile(32,70,198));
  svg = svg.replace(/<g transform="translate\(32,209\) scale\(1\)">[\s\S]*?<\/g>/, tile(32,204,291));
  svg = svg.replace('height="990" viewBox="0 0 1500 990"','height="1040" viewBox="0 0 1500 1040"');
  svg = svg.replace('<rect width="1500" height="990"','<rect width="1500" height="1040"');
  svg = svg.replace('Schematic design; region shapes, connections and damage maps are not experimental observations.','Candidate model: regions, support cards and output map are schematic, not measured results.');
  svg = svg.replace(/<\/svg>\s*$/, '<text x="25" y="1020" style="font-size:24px">Input examples: BRIGHT, Chen et al. (2025), Fig. 6, Turkey-EQ-2023; imagery credits: Table 2.</text></svg>');
  const out = path.join(dir,'BRIGHT_real_inputs_overview_v2');
  fs.writeFileSync(out+'.svg',svg);
  const browser = await chromium.launch({executablePath:'C:/Program Files/Google/Chrome/Application/chrome.exe',headless:true});
  const page = await browser.newPage({viewport:{width:1500,height:1040},deviceScaleFactor:2});
  await page.setContent('<html><body style="margin:0">'+svg+'</body></html>');
  await page.locator('svg').first().screenshot({path:out+'.png'});
  await browser.close();
  console.log(out);
})();
