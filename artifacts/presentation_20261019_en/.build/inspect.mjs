import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const p=await PresentationFile.importPptx(await FileBlob.load('/Users/lzy/Desktop/GitHub/seatrack/artifacts/presentation_20261016_final/output/SEATrack_论文复现汇报.pptx'));
const d='/Users/lzy/Desktop/GitHub/seatrack/artifacts/presentation_20261019_en/.build';
console.log((await p.inspect({kind:'slide,textbox,table,layout',maxChars:30000})).ndjson);
for(let i=0;i<p.slides.items.length;i++){
 const s=p.slides.items[i];
 await fs.writeFile(`${d}/before-${i+1}.png`,new Uint8Array(await (await p.export({slide:s,format:'png',scale:1})).arrayBuffer()));
 await fs.writeFile(`${d}/layout-${i+1}.json`,await (await s.export({format:'layout'})).text());
}
