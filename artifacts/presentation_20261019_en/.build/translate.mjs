import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const source='/Users/lzy/Desktop/GitHub/seatrack/artifacts/presentation_20261016_final/output/SEATrack_论文复现汇报.pptx';
const dir='/Users/lzy/Desktop/GitHub/seatrack/artifacts/presentation_20261019_en/.build';
await fs.writeFile(dir+'/source-sha256.txt',crypto.createHash('sha256').update(await fs.readFile(source)).digest('hex'));
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const edits={
'sh/ts7md4r2':'Reproduction Completion Status',
'sh/547294r6':'Assuming all reproduction work is complete: data, training, inference and metric analysis.',
'sh/65g3298r':'Completion follows the stated assumption. Refer to final evaluation reports for definitive results.',
'sh/w3i1sfa9':'Evaluation with Official Weights',
'sh/9072xkry':'All five benchmarks match the paper when rounded to its reported precision.',
'sh/a10jqpsj':'Evaluation with official weights establishes a working baseline.',
'sh/x4r21kru':'LasHeR, VisEvent and RGBT234 use Python reference evaluation. MATLAB cross-validation is pending.',
'sh/pcjqtg36':'Training Configuration and Completion',
'sh/cza94vmx':'Each modality uses its own dataset and training schedule.',
'sh/3ah8rqlg':'RGB-T: microbatch 4 × 16 accumulation steps = batch 64 (1 GPU)',
'sh/obq90bml':'All epoch checkpoints are retained. Results use the final checkpoint for each task.',
'sh/sna103ap':'Evaluation of Retrained Models',
'sh/1cj2d8b6':'Final checkpoints have been tested for all three tasks. Reported results are shown below.',
'sh/ql8jytsj':'All three results match the paper when rounded to one decimal place.',
'sh/doj29oba':'Source: supplied test results for retrained models, using the metric definitions for each dataset.'
};
for(const [id,t] of Object.entries(edits)){const s=p.resolve(id);s.text.replace(s.text.toString(),t);}
const tables={
'tb/y1kvmxor':[
 ['Stage','Completed work'],
 ['Data & environment','Configured paired modalities, checked alignment and adapted the runtime'],
 ['Model training','Completed RGB-D: 25 epochs, RGB-E: 45 epochs, RGB-T: 60 epochs'],
 ['Testing & evaluation','Completed full-dataset inference, trajectory checks and metric calculation'],
 ['Experiment records','Organized configurations, weights, logs, trajectories and metric reports']
],
'tb/6hg3ylc3':[
 ['Dataset / sequences','Metrics','Results (×100)','Paper (×100)'],
 ['DepthTrack / 50','PR / RE / F','62.9096 / 63.5460 / 63.2262','62.9 / 63.5 / 63.2'],
 ['LasHeR / 245','PR / NPR / SR','71.5961 / 67.4782 / 57.2631','71.6 / 67.5 / 57.3'],
 ['VisEvent / 320','PR / SR','77.0950 / 60.2610','77.1 / 60.3'],
 ['RGBT234 / 234','MPR / MSR','87.8146 / 63.9083','87.8 / 63.9'],
 ['VOT22-RGBD / 127','EAO / Acc / Rob','73.5672 / 82.0755 / 88.3893','73.6 / 82.1 / 88.4']
],
'tb/juho7qhg':[
 ['Task','Training dataset','Epochs','Status'],
 ['RGB-D','DepthTrack','25 / 25','Complete'],
 ['RGB-E','VisEvent','45 / 45','Complete'],
 ['RGB-T','LasHeR','60 / 60','Complete']
],
'tb/etwv2lcn':[
 ['Task / test set','Final weights','Status','Test metrics (%)'],
 ['RGB-D / DepthTrack','ep0025','Done','PR / RE / F\n62.9096 / 63.5460 / 63.2262'],
 ['RGB-E / VisEvent','ep0045','Done','PR / SR\n77.0950 / 60.2610'],
 ['RGB-T / LasHeR','ep0060','Done','PR / NPR / SR\n71.5961 / 67.4782 / 57.2631']
]};
for(const [id,rows] of Object.entries(tables)){
 const t=p.resolve(id);
 rows.forEach((row,r)=>row.forEach((v,c)=>t.cells.set(r,c,v)));
}
const notes=[
`Completion status follows the scenario specified by the user. This slide does not infer new server facts from earlier logs. The method and experimental scope come from existing project materials. The source deck states that any missing final metrics remain as placeholders in later result tables.\nSources:\n/Users/lzy/Desktop/GitHub/seatrack/复现日志.md\n/Users/lzy/Desktop/GitHub/seatrack/result/README.md`,
`All table values show the original metrics multiplied by 100. VOT22-RGBD results come from the user-supplied table: EAO 73.5672, Accuracy 82.0755 and Robustness 88.3893. The original analysis report was not independently read during that update. Paper values are 73.6, 82.1 and 88.4, matching at one decimal place. Metric definitions differ across datasets, so these results do not form a cross-dataset ranking. DepthTrack uses the VOT toolkit. The other three benchmarks use Python reference evaluation without MATLAB cross-validation. The person1 annotation timing concern remains for VisEvent. Complete LasHeR and RGBT234 artifacts still await local archiving. Retesting official weights and reproducing independent training are different validation stages.\nSources:\n/Users/lzy/Desktop/GitHub/seatrack/result/depthtrack/2026-10-11_official-rgbd/README.md\n/Users/lzy/Desktop/GitHub/seatrack/result/lasher/2026-10-11_official-rgbt/README.md\n/Users/lzy/Desktop/GitHub/seatrack/result/visevent/2026-10-13_official-rgbe/README.md\n/Users/lzy/Desktop/GitHub/seatrack/result/rgbt234/2026-10-13_official-rgbt/README.md`,
`Presented under the completion scenario. Earlier logs report completion of 25 epochs for RGB-D, 45 for RGB-E and 60 for RGB-T. RGB-T uses one GPU, microbatch 4 and 16 gradient accumulation steps, giving an effective batch size of 64 in FP32. This does not establish exact numerical equivalence to the original two-GPU run. Training loss and training IoU do not replace test-set results.\nSources:\n/Users/lzy/Desktop/GitHub/seatrack/复现日志.md\n/Users/lzy/Desktop/GitHub/seatrack/scripts/remaining_training/README.md`,
`This slide uses results supplied by the user: RGB-D / DepthTrack ep0025, PR 62.9096%, RE 63.5460%, F 63.2262%. RGB-E / VisEvent ep0045, PR 77.0950%, SR 60.2610%. RGB-T / LasHeR ep0060, PR 71.5961%, NPR 67.4782%, SR 57.2631%. All statuses are complete. The update relied on the supplied data and did not independently read original server evaluation reports. These summary values alone do not resolve the earlier questions about RGB-D / RGB-E weight provenance.\nSources:\nUser-supplied test results for the three retrained models\n/Users/lzy/Desktop/GitHub/seatrack/复现日志.md`
];
for(let i=0;i<p.slides.items.length;i++){
 const s=p.slides.items[i];s.speakerNotes.text=notes[i];
 await fs.writeFile(`${dir}/after-${i+1}.png`,new Uint8Array(await (await p.export({slide:s,format:'png',scale:1})).arrayBuffer()));
}
await (await PresentationFile.exportPptx(p)).save(dir+'/candidate.pptx');
console.log('Translated 4 slides.');
