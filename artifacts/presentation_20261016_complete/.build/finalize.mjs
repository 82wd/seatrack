import fs from 'node:fs/promises';
import { finalizePresentation } from '/Users/lzy/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations/container_tools/artifact_tool_utils.mjs';
const workspaceDir='/Users/lzy/Desktop/GitHub/seatrack/artifacts/presentation_20261016_complete';
const skill='/Users/lzy/.codex/plugins/cache/openai-primary-runtime/presentations/26.1013.11914/skills/presentations';
const {tableOwners}=JSON.parse(await fs.readFile(workspaceDir+'/.build/config.json','utf8'));
const result=await finalizePresentation({workspaceDir,candidatePath:workspaceDir+'/.build/candidate.pptx',finalPath:workspaceDir+'/output/SEATrack_论文复现完成汇报_20261016.pptx',pythonExecutable:'/Users/lzy/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3',integrityValidatorPath:skill+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:skill+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:tableOwners,fontPolicy:{basis:'design',families:['Hiragino Sans GB']},verifyArtifactToolImport:true,receiptPath:workspaceDir+'/.build/validation.json'});
console.log(JSON.stringify(result));
