import fs from 'node:fs/promises';
import path from 'node:path';
import { Presentation, PresentationFile } from '@oai/artifact-tool';
const ROOT='/Users/lzy/Desktop/GitHub/seatrack';
const WORK=ROOT+'/artifacts/presentation_20261016';
const BUILD=WORK+'/.build';
const SKILL='/Users/lzy/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const FONT='Hiragino Sans GB';
const P=Presentation.create({slideSize:{width:1280,height:720}});
const C={ink:'#182B40',blue:'#205CA0',teal:'#117E83',muted:'#637385',bg:'#F8FAFC',orange:'#BA5D23',line:'#D9E1E9'};
const sources={block:ROOT+'/SEATrack-main/lib/models/layers/attn_blocks.py',attn:ROOT+'/SEATrack-main/lib/models/layers/attn.py',vit:ROOT+'/SEATrack-main/lib/models/seatrack/vit_ci.py',head:ROOT+'/SEATrack-main/lib/models/layers/head.py',model:ROOT+'/SEATrack-main/lib/models/seatrack/seatrack.py',tracker:ROOT+'/SEATrack-main/lib/test/tracker/seatrack.py',log:ROOT+'/复现日志.md'};
const notes=[];const tableOwners=[];
function text(s,t,x,y,w,h,size=26,color=C.ink,bold=false){const q=s.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});q.text=t;q.text.style={typeface:FONT,fontSize:size,color,bold};return q;}
function slide(title,subtitle='',note='',refs=[]){const s=P.slides.add();s.background.fill=C.bg;text(s,title,64,44,1152,70,44,C.ink,true);if(subtitle)text(s,subtitle,66,120,1140,50,24,C.muted);text(s,String(P.slides.items.length).padStart(2,'0'),1170,664,50,28,18,C.muted);s.speakerNotes.text=note+'\n\n来源：\n'+refs.join('\n');notes.push({title,note,refs});return s;}
function foot(s,t,color=C.muted){text(s,t,66,624,1090,38,19,color);}
function para(s,label,body,y,{x=66,w=1128}={}){text(s,label,x,y,w,40,29,C.blue,true);text(s,body,x,y+48,w,90,26);}
function equation(s,t,y,size=34){text(s,t,76,y,1128,65,size,C.teal,true);}
function table(s,values,widths,y=190,h=340,size=25){const t=s.tables.add({rows:values.length,columns:values[0].length,left:66,top:y,width:1148,height:h,columnWidths:widths,values});t.borders.assign({fill:C.line,width:1,style:'solid'});for(let r=0;r<values.length;r++)for(let c=0;c<values[0].length;c++){const cell=t.getCell(r,c);cell.fill=r===0?C.ink:(r%2?'#FFFFFF':'#EEF3F7');cell.text.style={typeface:FONT,fontSize:size,color:r===0?'#FFFFFF':C.ink,bold:r===0};}tableOwners.push(P.slides.items.length);return t;}
// 01
{
const s=slide('SEATrack 核心机制与复现进展','', '本次汇报分为方法理解和复现进度两部分。以一个RGB与热红外样本说明数据在AMG-LoRA、HMoE和预测头中的流动。计算数值用于教学，实验数值来自仓库记录。进度证据最近更新为2026年10月14日，没有在制作时连接服务器查询。',[ROOT+'/code.md',sources.log]);
s.background.fill='#12273C';s.shapes.items[0].text.style={typeface:FONT,fontSize:48,color:'#FFFFFF',bold:true};text(s,'AMG-LoRA · HMoE · Prediction Head',68,202,1130,60,34,'#84D3D4');text(s,'一个样本的计算过程\n官方权重评测、自训练与待完成验证',68,330,1100,120,34,'#FFFFFF');text(s,'汇报整理：2026年10月16日\n进度依据：截至10月14日的本地记录',70,562,1080,70,23,'#BBCAD7');
}
//02
{
const s=slide('模型结构与本次汇报范围','两路特征经过共享骨干，增强层加入注意力引导与跨模态融合。','图为仓库提供的论文框架图。默认骨干12层，零起始索引1、3、5、7、9、11使用AMG-LoRA和两处HMoE。每层两模态共享该层参数，层与层之间参数独立。Prediction Head位于Encoder之后。',[ROOT+'/SEATrack-main/assets/Framework.png',sources.vit,sources.block]);
const b=await fs.readFile(ROOT+'/SEATrack-main/assets/Framework.png');s.images.add({blob:b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength),contentType:'image/png',alt:'SEATrack论文框架图',fit:'contain',position:{left:65,top:182,width:1150,height:420}});foot(s,'12 个 block；第 2、4、6、8、10、12 层包含增强模块。');
}
//03
{
const s=slide('贯穿全程的单样本','B = 1，RGB 与热红外各有一张模板和一张搜索图。','模板来自目标参考区域，搜索来自当前帧目标附近。输入尺寸和token数量来自默认配置。搜索网格第7行第9列按零起始编号为121，在模板搜索拼接后的序列中编号185。这个位置只是教学跟踪位置，并不预先认定其为真实目标中心。',[sources.vit,ROOT+'/SEATrack-main/lib/models/layers/patch_embed.py']);
table(s,[['输入','图像尺寸','Token 形状'],['RGB / 红外模板，各一张','[1,3,128,128]','各 [1,64,768]'],['RGB / 红外搜索，各一张','[1,3,256,256]','各 [1,256,768]'],['每路模板与搜索拼接','64 + 256 个位置','各 [1,320,768]']],[410,360,378],192,280,26);
text(s,'持续跟踪：搜索网格 (行7，列9)',76,502,1100,42,30,C.blue,true);text(s,'搜索编号 121 = 7 × 16 + 9；完整序列编号 185 = 64 + 121',76,555,1100,40,26);foot(s,'尺寸来自源码；后续局部数字均为教学示例，非 checkpoint 实测输出。');
}
//04
{
const s=slide('AMG-LoRA：Q、K、V 的生成','自定义 MergedLinear 组织联合投影，仅对 K、V 增加低秩更新。','先执行库中的LayerNorm。公式使用行向量右乘写法，实际PyTorch权重存储为out,in。省略训练dropout。r=8，K和V拥有各自的A和B。源码定义了alpha/r但注释了实际乘法。推理时可以合并LoRA增量，等价于这里的展开表达。',[sources.attn,sources.block]);
equation(s,'Q = H Wq + bq',195);equation(s,'K = H Wk + bk + H Ak Bk',275);equation(s,'V = H Wv + bv + H Av Bv',355);
text(s,'A：768 × 8      B：8 × 768',78,455,1110,45,30,C.blue,true);text(s,'每个投影新增 12,288 个参数；Q 不加 LoRA。\n联合输出 [1,320,2304]，拆成 12 个头后 Q/K/V 各为 [1,12,320,64]。',78,518,1120,90,25);foot(s,'调用路径：cal_qkv() 内部直接调用 self.attn.qkv()。');
}
//05
{
const s=slide('AMG：两种模态互相引导注意力分数','先计算各自的 QKᵀ，再在 softmax 之前混合分数。','S是softmax前的logits，P是softmax后的权重。两路都引用对方原始logits。α对应dte2r_scaling，β对应r2dte_scaling，都是每增强层一个方向一个可学习标量，初始为1，无sigmoid约束。下方仅用两个key演示，实际每行有320个key。',[sources.block]);
equation(s,'S = QKᵀ / √64     [1,12,320,320]',185,32);equation(s,'S′RGB = SRGB + α(SX − SRGB)',265,32);equation(s,'S′X = SX + β(SRGB − SX)',335,32);
text(s,'两位置演算',78,432,450,40,29,C.blue,true);text(s,'SRGB = [2,0]，SX = [0,2]，α = 0.75\nS′RGB = [0.5,1.5]\nsoftmax(S′RGB) ≈ [0.269,0.731]',78,482,1100,125,27);foot(s,'教学示例；引导系数随训练更新，推理时不由每帧重新预测。');
}
//06
{
const s=slide('注意力输出：按新权重读取本模态 V','对方模态影响关注位置，本模态 V 提供被汇总的内容。','以下两个通道仅是教学缩小例子。真实计算每个头64维，12个头合并后做输出投影，返回分支更新量。xrgb_attn变量名虽然包含attn，但这里已经是token特征。残差保留LayerNorm之前的原输入，推理DropPath关闭。',[sources.block,sources.attn]);
equation(s,'Oᵢ = Σⱼ Pᵢⱼ Vⱼ',192,40);
text(s,'V₁ = [1,2]，V₂ = [3,0]',78,282,1100,44,31);text(s,'0.269 V₁ + 0.731 V₂ = [2.462,0.538]',78,344,1100,55,34,C.teal,true);
table(s,[['阶段','形状'],['每个头加权汇总 V','[1,12,320,64]'],['合并头并做输出投影','[1,320,768]'],['原特征 + 注意力更新量','两路各 [1,320,768]']],[700,448],427,176,23);foot(s,'RGB 与红外分别更新；模板与搜索已经在 320 × 320 的注意力中交互。');
}
//07
{
const s=slide('HMoE：模板、搜索分别跨模态融合','自定义 HMoE 接收两种模态的特征内容，生成每个位置的更新量。','第一处attn_moe有4个专家，每个专家2个槽位，低秩维4。模板和搜索调用同一个模块、共享参数，mode参数当前未参与内部计算。跟踪位置在搜索融合输入中的RGB编号为121，红外编号为377。',[sources.block,sources.attn]);
table(s,[['分组','拼接顺序','输入形状'],['模板','RGB 64，红外 64','[1,128,768]'],['搜索','RGB 256，红外 256','[1,512,768]']],[210,530,408],193,230,27);
para(s,'搜索组中，同一空间位置有两个输入','RGB 编号 121；红外编号 377 = 256 + 121。\n第一处 HMoE 使用 4 个专家、每专家 2 个槽位。',465);foot(s,'Divide & Concat 改变分组和排列；后面会拆回原来的两路序列。');
}
//08
{
const s=slide('HMoE Dispatch：从输入汇聚到槽位','LayerNorm 与低秩映射后，每个 token 拆成两个 384 维子向量。','LoRP的实际式为(LN(X)A+a)B+b，768到4到768，默认无GELU。输入搜索组N=512，拆分后U为1024×384。Phi为384×8。Dispatch沿输入子token维softmax，每个槽位的输入权重和为1。温度TD为可学习参数，初值1，未约束为正。',[sources.attn]);
equation(s,'U = reshape(LoRP(LN(X)))     [1,1024,384]',190,30);equation(s,'L = UΦ     Φ：[384,8]     L：[1,1024,8]',264,30);equation(s,'Dₐₖ = exp(Lₐₖ / TD) / Σₐ′ exp(Lₐ′ₖ / TD)',339,30);equation(s,'Gₖ = Σₐ Dₐₖ Uₐ',414,34);
text(s,'每个槽位读取所有输入子向量的加权和。\nRGB 与红外的信息在这里共同汇入槽位，输出整理为 [1,4,2,384]。',78,514,1100,90,26);foot(s,'第一处 HMoE：4 个专家 × 2 个槽位，共 8 个汇聚位置。');
}
//09
{
const s=slide('HMoE：专家变换与 Combine','每个专家处理汇聚特征，每个 token 再按自己的权重组合专家结果。','专家计算Ye,s=(Ge,s Ae+ae)Be+be，384到4到384，当前无GELU。不同专家参数独立，同一专家两个槽位共享参数。拼接两槽位为768维后经过linear2的LoRP。Combine来自同一原始logits，对每个token的2个子向量与每专家2个槽位的4项求和，再沿专家维softmax。所有专家参与，无top-k筛选。',[sources.attn]);
text(s,'专家处理',76,190,1100,42,29,C.blue,true);text(s,'每槽位 384 / 4 / 384 维低秩变换\n每专家拼接两个槽位，再经 LoRP，得到 F：[1,4,768]',76,241,1110,90,27);
equation(s,'Rᵢₑ = Σₐ₌₁² Σₛ₌₁² L₍ᵢ,ₐ₎,₍ₑ,ₛ₎',360,31);equation(s,'C = softmax(R / TC，沿专家维)     O = CF',435,30);
text(s,'C：[1,512,4]      F：[1,4,768]      O：[1,512,768]',76,531,1120,48,28,C.blue,true);foot(s,'Dispatch 对输入归一化；Combine 对专家归一化，二者来自同一份 logits。');
}
//10
{
const s=slide('HMoE 的单通道演算与拆回','同一组专家结果可以通过不同权重，生成不同 token 的更新量。','示例权重和值均非实测。RGB搜索token121的某通道更新为1.6，原值0.7时残差结果2.3。模板输入也独立完成相同过程，之后按各模态模板搜索顺序重组。随后MLP和第二处HMoE继续更新，第二处8专家，参数独立。',[sources.block,sources.attn]);
table(s,[['专家','1','2','3','4'],['本 token 的 Combine','0.5','0.2','0.2','0.1'],['专家输出的某一通道','2','4','−1','0']],[408,185,185,185,185],190,220,25);
equation(s,'更新量 = 0.5×2 + 0.2×4 + 0.2×(−1) = 1.6',444,29);
text(s,'若原特征该通道为 0.7，残差相加后为 2.3。\n拆回 RGB 与红外，重组各自模板和搜索，每路仍是 [1,320,768]。',78,523,1120,85,25);foot(s,'随后经过 MLP 和第二处 8 专家 HMoE，再继续后续 block。');
}
//11
{
const s=slide('Prediction Head：搜索特征的三个分支','Encoder 最终归一化后，取两路搜索 token 相加并恢复 16 × 16 网格。','CenterPredictor在本地项目中定义，内部卷积BN/ReLU来自PyTorch。最终只保留搜索，模板信息已融入搜索。每个分支四层3×3 Conv-BN-ReLU，通道768、256、128、64、32，最后1×1卷积。中心和尺寸sigmoid后clamp，偏移保持实数。',[sources.vit,sources.model,sources.head]);
equation(s,'F = HRGB,search + HX,search     [1,256,768]',185,30);
text(s,'恢复空间排列：[1,768,16,16]；编号 121 回到第 7 行、第 9 列。',78,260,1120,60,26);
table(s,[['分支','输出形状','作用'],['中心分数','[1,1,16,16]','选择目标中心位置'],['尺寸','[1,2,16,16]','预测归一化宽、高'],['偏移','[1,2,16,16]','修正网格中心坐标']],[285,360,503],350,238,26);foot(s,'三个分支参数独立；卷积读取空间邻域，分数与尺寸使用 sigmoid。');
}
//12
{
const s=slide('预测头解码：由同一位置取中心、尺寸和偏移','假设中心分数最高点为第 7 行、第 9 列，分数 0.92。','教学示例：idx=121，offset=[0.25,0.50]，size=[0.25,0.50]。先从中心图找最大位置，再在同一个idx读取size和offset，不能三个分支各取最大。尺寸已相对整个搜索图归一化，不再除16。输出顺序cxcywh。',[sources.head]);
table(s,[['该位置的输出','数值'],['网格坐标 (x,y)','(9,7)'],['偏移 (dx,dy)','(0.25,0.50)'],['尺寸 (w,h)','(0.25,0.50)']],[660,488],188,250,27);
equation(s,'cx = (9 + 0.25)/16 = 0.578125',465,31);equation(s,'cy = (7 + 0.50)/16 = 0.46875',525,31);foot(s,'归一化框：[0.578125, 0.46875, 0.25, 0.50]，顺序为 cx、cy、w、h。');
}
//13
{
const s=slide('目标框映射回原图','先恢复尺度，再平移到原图，最后转换为左上角与宽高。','搜索图尺寸256，原裁剪边长512，resize_factor=0.5。假设代码使用的裁剪原点为(100,200)。Tracker先用Hann窗口调整中心分数并再次解码，此例假设赢家仍为7,9。归一化框乘512得到局部中心296,240、宽高128,256。加原点得到396,440，再减半宽高得到332,312。最终clip_box需结合原图尺寸。',[sources.tracker,sources.head]);
table(s,[['坐标阶段','中心 (cx,cy)','宽高 (w,h)'],['256 × 256 搜索图','(148,120)','(64,128)'],['原裁剪区域，边长512','(296,240)','(128,256)'],['加原点 (100,200)','(396,440)','(128,256)']],[500,324,324],190,290,26);
equation(s,'最终 xywh = [332,312,128,256]',518,37);foot(s,'实际 Tracker 先乘 Hann 窗再解码；本例假设选中位置不变，最后还会裁剪边界。');
}
//14
{
const s=slide('论文复现进度总览','已形成官方权重评测结果，自训练验证仍有未完成事项。','本页依据本地复现日志最新记录至2026-10-14。result/README.md对VOT22仍是旧状态，因此采用较新的详细日志。制作于2026-10-16，不宣称实时状态。三个训练任务均报告完成，但验证证据强度不同。',[sources.log,ROOT+'/result/README.md']);
table(s,[['工作项','最新有证据的状态'],['官方权重评测','DepthTrack、LasHeR、VisEvent、RGBT234 已有指标'],['三项自训练','RGB-D 25轮、RGB-E 45轮、RGB-T 60轮均报告完成'],['自训练测试','RGB-D、RGB-E 已有结果；RGB-T 入口已准备'],['VOT22 官方权重测试','10月14日确认运行，尚无全量完成和最终指标记录']],[350,798],191,354,25);foot(s,'进度依据截至 2026-10-14；汇报整理于 2026-10-16，未进行实时服务器查询。',C.orange);
}
//15
{
const refs=['depthtrack/2026-10-11_official-rgbd','lasher/2026-10-11_official-rgbt','visevent/2026-10-13_official-rgbe','rgbt234/2026-10-13_official-rgbt'].map(v=>ROOT+'/result/'+v+'/README.md');
const s=slide('四项官方权重评测结果','按论文展示精度舍入后，各项指标与论文记录一致。','所有表中数值为百分数。不同数据集指标定义不同，不做横向排名。DepthTrack使用VOT工具分析。其余三项为Python参考口径，未执行MATLAB交叉验证。VisEvent保留person1标注时间对应风险。LasHeR与RGBT234完整产物尚待本地归档。官方权重复测与独立训练复现是不同验证阶段。',refs);
table(s,[['数据集 / 序列数','指标','实测（%）','论文（%）'],['DepthTrack / 50','PR / RE / F','62.9096 / 63.5460 / 63.2262','62.9 / 63.5 / 63.2'],['LasHeR / 245','PR / NPR / SR','71.5961 / 67.4782 / 57.2631','71.6 / 67.5 / 57.3'],['VisEvent / 320','PR / SR','77.0950 / 60.2610','77.1 / 60.3'],['RGBT234 / 234','MPR / MSR','87.8146 / 63.9083','87.8 / 63.9']],[275,220,398,255],190,350,22);
text(s,'官方权重评测已经建立可用基线。',76,564,1100,40,29,C.teal,true);foot(s,'LasHeR、VisEvent、RGBT234 使用 Python 参考口径；尚未做 MATLAB 交叉验证。');
}
//16
{
const s=slide('三项训练均已报告完成','完成训练与获得独立可信的测试结论，需要分别核验。','RGB-D第5到25轮234个张量变化，优化器步数4685到23425。RGB-E45轮，早末轮228个张量变化。RGB-T完成状态来自用户回传面板，本次没有独立读取completion.json或逐个检查checkpoint内容。RGB-T微批4累积16保持有效batch64，但不能宣称与原双卡训练数值完全等价。训练IoU和loss不能当作测试集指标。',[sources.log,ROOT+'/scripts/remaining_training/README.md']);
table(s,[['任务','完成轮次','已有证据'],['RGB-D / DepthTrack','25 / 25','末轮权重存在；第5至25轮参数与优化器步数变化'],['RGB-E / VisEvent','45 / 45','45份 checkpoint；早末轮228个张量变化'],['RGB-T / LasHeR','60 / 60','用户面板报告完成；60份 checkpoint，内容待独立复核']],[310,180,658],190,300,25);
text(s,'RGB-T 使用单卡微批 4 × 累积 16，有效 batch = 64。',76,531,1130,44,27,C.blue,true);foot(s,'训练完成状态不等于测试达标；checkpoint 数量不替代逐文件内容验证。');
}
//17
{
const s=slide('自训练评测与权重来源问题','RGB-D、RGB-E 的测试结果已记录，但独立复现结论仍需来源证据。','RGB-D已有PR62.9096/RE63.5460/F63.2262，与当前官方路径模型参数相同，来源疑点未解决。RGB-E自训练320条跟踪完成，PR20=77.09504076887207和SR_AUC=60.2609637859865，逐条轨迹相同、不同0、缺失0；当前官方路径和自训练末轮net参数逐张量相同，文件不是samefile。不能据此推断无训练更新，也不能据此证明独立训练精确复现。用户已要求暂缓核查并优先进行RGB-T测试。',[sources.log]);
table(s,[['任务','评测记录','当前限制'],['RGB-D','DepthTrack PR / RE / F\n62.9096 / 63.5460 / 63.2262','与当前官方路径模型参数相同\n来源尚未厘清'],['RGB-E','VisEvent PR / SR\n77.0950 / 60.2610','320条轨迹及模型参数相同\n来源核查暂缓'],['RGB-T','末轮 ep0060 测试入口已准备','尚无测试完成或指标回传记录']],[185,480,483],185,365,24);foot(s,'结论边界：汇总指标一致，尚不足以证明独立训练复现成功。',C.orange);
}
//18
{
const s=slide('VOT22：最近记录仍处于全量评测阶段','以下是 2026年10月14日采样快照，不代表今天的进度。','数据在本地与服务器均通过127序列、80741原始帧结构验收。单序列adapter01_indoor_1成功后启动全量。最新面板23/127序列，356/1765起点，对应283047/1327004重复跟踪帧。多起点协议使跟踪帧数大于原始数据帧数。该快照不是耗时比例。需要全量退出码、轨迹完整性和官方分析指标才能确认完成。',[sources.log,ROOT+'/scripts/remaining_official/vot22_progress.py']);
table(s,[['检查项','记录'],['数据准备','127 序列、80,741 原始帧，服务器结构验收通过'],['单序列试跑','adapter01_indoor_1 成功，退出码 0'],['已完成序列','23 / 127，18.11%'],['已完成起点','356 / 1,765，20.17%'],['最终指标','尚无记录']],[350,798],190,360,25);foot(s,'待补：最新状态、全量退出码、结果验收和官方指标分析。',C.orange);
}
//19
{
const s=slide('已完成的工程工作与评测边界','数据、运行与结果验收已有流程，仍需保留口径和来源限制。','源码讲解与复现工作结合：加入路径隔离、参数初始化校验、独立结果目录、锁、轨迹检查和指标脚本。RGBT234修复换行兼容只处理内存文本，不修改原始标注。VisEvent特殊person1序列原始332帧与官方标注306帧，按官方参考程序取预测前缀，并保留风险。不同数据集指标口径遵循各自参考规则。',[sources.log,ROOT+'/scripts/lasher_metrics.md',ROOT+'/result/visevent/2026-10-13_official-rgbe/README.md']);
para(s,'训练与推理隔离','独立训练副本、独立结果目录、显式末轮权重路径，记录 SHA-256。',185);
para(s,'数据与轨迹验收','检查序列清单、帧数、双模态配对、预测有限值和正宽高。',330);
para(s,'协议差异持续留档','VisEvent 一条序列标注对应问题保留；三项 Python 指标仍待 MATLAB 对照。',475);foot(s,'LasHeR、RGBT234 已有回传摘要，完整报告与轨迹还需归档。');
}
//20
{
const s=slide('下一步工作','先补齐已安排的评测，再按约定恢复权重来源核查。','这是待执行计划，不代表已经运行。遵循用户此前暂缓RGB-E来源排查、先继续RGB-T评测的顺序。RGB-T入口已写好但尚无服务器运行完成证据。VOT22只读确认现有任务，避免重复启动。来源核查需要比对原始下载、官方目标、自训练末轮SHA，再审查初始化和恢复记录。',[sources.log]);
table(s,[['顺序','具体工作','完成证据'],['1','RGB-T 末轮 LasHeR 测试','单序列试跑、245序列轨迹验收、PR/NPR/SR'],['2','补齐 VOT22 全量评测记录','最新状态、退出码、全起点结果及官方指标'],['3','恢复 RGB-D / RGB-E 来源核查','下载源与两套权重 SHA、参数和恢复链路'],['4','统一归档与协议对照','配置、日志、指标、轨迹及校验清单']],[100,440,608],190,350,25);foot(s,'独立复现的最终结论需要训练来源、测试协议和原始结果形成完整证据链。');
}
//21
{
const s=slide('附录：源码与复现记录入口','模块公式与项目实现对应，详细路径和讲稿见每页备注。','模块来自项目本地实现，不宣称所有算法皆由作者原创。cal_qkv/amglora_attn、HMoE/LoRP/MultiExpertLinear、CenterPredictor均为项目定义；nn.Conv2d、LayerNorm、softmax、矩阵运算由库提供。参考本地code.md、复现日志、各结果README及核心源码。未在线访问论文网页。',[sources.block,sources.attn,sources.head,sources.vit,sources.model,sources.tracker,sources.log,ROOT+'/code.md']);
table(s,[['模块 / 记录','入口'],['AMG-LoRA','attn_blocks.py：cal_qkv / amglora_attn\nattn.py：MergedLinear'],['HMoE','attn.py：HMoE / LoRP / MultiExpertLinear'],['Prediction Head','head.py：CenterPredictor / cal_bbox\nseatrack.py：forward_head / map_box_back'],['复现进度与结果','复现日志.md\nresult/<数据集>/<实验>/README.md']],[330,818],190,372,25);foot(s,'所有局部计算数字均用于说明原理；实验指标单独标明来源与验证范围。');
}
await fs.writeFile(BUILD+'/speaker-notes.json',JSON.stringify(notes,null,2));
await fs.writeFile(BUILD+'/讲稿.md',notes.map((n,i)=>`## ${i+1}. ${n.title}\n\n${n.note}\n\n来源：\n${n.refs.join('\n')}\n`).join('\n'));
await fs.mkdir(BUILD+'/previews',{recursive:true});
for(let i=0;i<P.slides.items.length;i++){const png=await P.export({slide:P.slides.items[i],format:'png',scale:1});await fs.writeFile(BUILD+'/previews/'+String(i+1).padStart(2,'0')+'.png',new Uint8Array(await png.arrayBuffer()));console.log('rendered',i+1);}
await (await PresentationFile.exportPptx(P)).save(BUILD+'/candidate.pptx');
await fs.writeFile(BUILD+'/config.json',JSON.stringify({tableOwners}));
console.log('DRAFT_READY');
