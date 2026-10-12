#!/usr/bin/env bash
# SEATrack 数据集服务器直连下载脚本
# 所有下载源已于 2026-10 实测：
#   DepthTrack  : Zenodo 官方记录（按序列拆分成多个 zip）
#   VOT22-RGBD  : VOT 官方数据站 description.json -> annotation/color/depth 三个 zip/序列
#   LasHeR      : HF 镜像 xche32/lasher（5 个分包，解压后 ~220GB）
#   VisEvent    : HF 镜像 krisspy39/visevent（官方 VictEvent_train/test 分包 zip，~232GB）或 xche32/visevent（单包 69GB）
# 用法见 ./dl.sh help

set -euo pipefail
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# ---------- 路径（可用环境变量覆盖） ----------
ROOT=${ROOT:-/nas/liziyan103/seatrack}
CODE=${CODE:-$ROOT/SEATrack-main}
DATA=${DATA:-$ROOT/datasets}
DL=${DL:-$ROOT/downloads}
HF_HOST=${HF_HOST:-https://hf-mirror.com}      # 若服务器能连 huggingface.co，可改成 https://huggingface.co
JOBS=${JOBS:-4}                                 # 并发下载数

ZENODO=https://zenodo.org/records
VOT_BASE=https://data.votchallenge.net/vot2022/rgbd

log()  { echo -e "\033[1;36m[dl]\033[0m $*"; }
warn() { echo -e "\033[1;33m[dl]\033[0m $*"; }
err()  { echo -e "\033[1;31m[dl]\033[0m $*" >&2; }
die()  { err "$*"; exit 1; }

usage() {
cat <<EOF
用法: bash dl.sh <命令> [参数]

  check                        检查依赖、磁盘与目标路径
  depthtrack test|train|all    下载 DepthTrack（test 18.3GB / train 53.8GB）
  votrgbd [N]                  下载 VOT22-RGBD 全部 127 序列（20.7GB），N 为只下前 N 个（试跑）
  lasher test|train|all        下载 LasHeR（test ~59GB / train ~161GB，需 ≥450GB 磁盘）
  visevent <src> test|train|all 下载 VisEvent，src 为 krisspy（~232GB，官方分包）或 xche32（69GB）
  verify                       校验各数据集序列数与目录结构

环境变量：ROOT CODE DATA DL HF_HOST JOBS
示例：
  bash dl.sh check
  JOBS=8 bash dl.sh depthtrack test
  tmux new -s dl && bash dl.sh lasher test
EOF
}

# ---------- 通用下载（单文件，支持断点续传，供 xargs 并行调用） ----------
cmd__fetch() {            # 隐藏子命令： dl.sh _fetch <url> <outfile>
  local url="$1" out="$2" i=1
  mkdir -p "$(dirname "$out")"
  if [ -s "$out" ]; then return 0; fi
  log "→ $(basename "$out")"
  for i in 1 2 3 4 5; do
    if command -v wget >/dev/null 2>&1; then
      wget -c -q --tries=5 --waitretry=10 --timeout=60 -O "$out" "$url" && return 0
    else
      curl -L -C - --retry 5 --retry-delay 5 -sS --max-time 3600 -o "$out" "$url" && return 0
    fi
    warn "第 ${i} 次失败，重试: $(basename "$out")"
    sleep 5
  done
  err "下载失败: $url"
  return 1
}

parallell_fetch() {       # $1 = plan 文件，每行 "<url> <outfile>"（用空格分隔，无空格路径）
  local plan="$1"
  [ -s "$plan" ] || die "计划文件为空: $plan"
  local n; n=$(wc -l < "$plan" | tr -d ' ')
  log "开始下载 ${n} 个文件, 并发 ${JOBS}"
  xargs -P "$JOBS" -n 2 -r bash "$SELF" _fetch < "$plan"
}

flatten_one() {           # 若 zip 解出同名嵌套目录，压平一层
  local outer="$1"
  local inner
  for inner in "$outer"/*; do
    [ -d "$inner" ] || continue
    if [ "$(basename "$inner")" = "$(basename "$outer")" ]; then
      mv "$inner"/* "$outer"/ 2>/dev/null || true
      rmdir "$inner" 2>/dev/null || true
    fi
  done
}

# ---------- 环境体检 ----------
cmd_check() {
  local miss=0
  for t in python3 unzip tar; do
    command -v $t >/dev/null 2>&1 || { err "缺少 $t"; miss=1; }
  done
  command -v wget >/dev/null 2>&1 || command -v curl >/dev/null 2>&1 || { err "wget/curl 至少要有一个"; miss=1; }
  command -v zip >/dev/null 2>&1 || warn "没装 zip：VisEvent 分包合并需要它（apt install zip / conda install zip）"
  command -v 7z  >/dev/null 2>&1 || warn "没装 7z：VisEvent 分包解压备选方案"
  [ "$miss" = 0 ] || exit 1
  log "依赖 OK"
  echo "  ROOT=$ROOT"
  echo "  CODE=$CODE"
  echo "  DATA=$DATA"
  echo "  DL  =$DL"
  echo "  HF  =$HF_HOST"
  df -h "$ROOT" | tail -2
  log "连通性："
  curl -s -o /dev/null -w "  zenodo %{http_code}\n"   -m 20 "$ZENODO/5792146" || true
  curl -s -o /dev/null -w "  vot    %{http_code}\n"   -m 20 "$VOT_BASE/description.json" || true
  curl -s -o /dev/null -w "  hf     %{http_code}\n"   -m 20 "$HF_HOST/datasets/xche32/lasher/resolve/main/lasher.tar.gz.part.aa" || true
}

# ---------- DepthTrack（RGB-D） ----------
cmd_depthtrack() {
  local split="${1:-test}"
  case "$split" in
    test)  set -- 5792146 ;;
    train) set -- 5794115 5837926 ;;
    all)   set -- 5792146 5794115 5837926 ;;
    *) die "用法: depthtrack test|train|all" ;;
  esac

  for rec in "$@"; do
    local outdir="$DL/depthtrack/$rec" outroot
    mkdir -p "$outdir"
    log "拉取 Zenodo 记录 $rec 的文件清单"
    curl -s --retry 5 --max-time 120 "$ZENODO/$rec" \
      | python3 -c "import sys,json; d=json.load(sys.stdin); print('  记录:', d['metadata']['title'], '文件数:', len(d['files']), '合计 %.2f GB' % (sum(f['size'] for f in d['files'])/1e9))"
    curl -s --retry 5 --max-time 120 "$ZENODO/$rec" \
      | python3 -c "import sys,json; [print(f['key']) for f in json.load(sys.stdin)['files']]" > "$outdir/filelist.txt"

    awk -v base="$ZENODO/$rec/files" '{print base "/" $0 "?download=1" "\t" $0}' "$outdir/filelist.txt" > "$outdir/plan.txt"
    ( cd "$outdir" && xargs -P "$JOBS" -n 2 -r bash "$SELF" _fetch < plan.txt )

    # 是每个 record 对应一个 split
    case "$rec" in
      5792146) outroot="$DATA/depthtrack/testingset" ;;
      *)       outroot="$DATA/depthtrack/trainingset" ;;
    esac
    mkdir -p "$outroot"
    log "解压到 $outroot"
    for z in "$outdir"/*.zip; do
      n="$(basename "${z%.zip}")"
      mkdir -p "$outroot/$n"
      unzip -q -o "$z" -d "$outroot/$n"
      flatten_one "$outroot/$n"
    done
    log "记录 $rec 完成，当前序列数：$(ls -1 "$outroot" | wc -l)"
  done

  echo
  cmd_verify depthtrack
}

# ---------- VOT22-RGBD（RGB-D） ----------
cmd_votrgbd() {
  local limit="${1:-0}"
  local seqdir="${SEQDIR:-$CODE/VOT22RGBD_workspace/sequences}"
  local work="$DL/votrgbd"
  mkdir -p "$work" "$seqdir"

  log "读取 VOT 官方数据集描述并生成下载计划"
  python3 - "$VOT_BASE" "$limit" "$work" <<'PY'
import json, sys, urllib.request, posixpath
base, limit, work = sys.argv[1], int(sys.argv[2]), sys.argv[3]
d = json.load(urllib.request.urlopen(base + '/description.json', timeout=120))
seqs = d['sequences']
print('  官方序列数:', len(seqs))
if limit > 0:
    seqs = seqs[:limit]
rows = []
for s in seqs:
    n = s['name']
    ann = base + '/' + s['annotations']['url']
    def fix(u):
        return u.replace('../../', 'https://data.votchallenge.net/')
    col = fix(s['channels']['color']['url'])
    dep = fix(s['channels']['depth']['url'])
    rows.append((n, ann, col, dep))
with open(work + '/table.tsv', 'w') as f:
    for n, a, c, dp in rows:
        f.write('\t'.join([n, a, c, dp]) + '\n')
plan = open(work + '/plan.txt', 'w')
for n, a, c, dp in rows:
    plan.write(f"{a}\t{work}/{n}/ann.zip\n")
    plan.write(f"{c}\t{work}/{n}/color.zip\n")
    plan.write(f"{dp}\t{work}/{n}/depth.zip\n")
plan.close()
print('  计划下载 zip 数:', len(rows) * 3)
PY

  parallell_fetch "$work/plan.txt"

  log "解压到 $seqdir"
  while IFS=$'\t' read -r n _ _ _; do
    [ -n "$n" ] || continue
    mkdir -p "$seqdir/$n/color" "$seqdir/$n/depth"
    unzip -q -o "$work/$n/ann.zip"   -d "$seqdir/$n"
    unzip -q -o "$work/$n/color.zip" -d "$seqdir/$n/color"
    unzip -q -o "$work/$n/depth.zip" -d "$seqdir/$n/depth"
    log "  ✓ $n ($(ls -1 "$seqdir/$n/color" | wc -l) 帧)"
  done < "$work/table.tsv"
  echo
  cmd_verify votrgbd
}

# ---------- LasHeR（RGB-T） ----------
cmd_lasher() {
  local split="${1:-test}"
  case "$split" in
    test)  local pats="testingset" ;;
    train) local pats="trainingset" ;;
    all)   local pats="trainingset testingset" ;;
    *) die "用法: lasher test|train|all" ;;
  esac

  local d="$DL/lasher"
  mkdir -p "$d"
  local plan="$d/plan.txt"; : > "$plan"
  for p in aa ab ac ad ae; do
    local f="lasher.tar.gz.part.$p"
    for host in "$HF_HOST" https://huggingface.co https://hf-mirror.com; do
      echo "$host/datasets/xche32/lasher/resolve/main/$f $d/$f" >> "$plan"
    done
  done
  warn "分包共 5 个（~224GB），同名的已存在文件会自动跳过；可用 JOBS 调并发"
  # 每个分包只在第一个可用源上成功，后续重复行因文件已存在会秒跳过
  xargs -P 1 -n 2 -r bash "$SELF" _fetch < "$plan"

  if [ ! -s "$d/lasher.tar.gz" ]; then
    log "合并分包 -> lasher.tar.gz"
    cat "$d"/lasher.tar.gz.part.a{a,b,c,d,e} > "$d/lasher.tar.gz"
  fi
  ls -lh "$d/lasher.tar.gz"

  local x="$d/x"
  mkdir -p "$x"
  for pat in $pats; do
    log "解出 ${pat}, 会花很久, 建议放 tmux 里跑"
    tar -xzf "$d/lasher.tar.gz" -C "$x" --wildcards "*/$pat/*" || warn "tar 返回非 0，检查是否磁盘已满"
    # 把解出来的 xxx/testingset 挪到 $DATA/lasher/ 下
    local found
    found=$(find "$x" -maxdepth 3 -type d -name "$pat" | head -1)
    [ -n "$found" ] || die "没找到 $pat 目录，先在 lasher.tar.gz 里确认实际层级： tar -tzf ... | head"
    mkdir -p "$DATA/lasher/$pat"
    # 用 cp -al（硬链接，省磁盘）失败则普通移动
    cp -al "$found"/. "$DATA/lasher/$pat"/ 2>/dev/null || cp -a "$found"/. "$DATA/lasher/$pat"/
    log "  ${pat} 序列数: $(ls -1 "$DATA/lasher/$pat" | wc -l)"
  done
  warn "解压完成。确认无误后可释放空间： rm -f $d/lasher.tar.gz $d/lasher.tar.gz.part.* $d/x -r"
  echo
  cmd_verify lasher
}

# ---------- VisEvent（RGB-E，很大） ----------
cmd_visevent() {
  local src="${1:-krisspy}" split="${2:-test}"
  case "$split" in
    test)  local sets="test" ;;
    train) local sets="train" ;;
    all)   local sets="train test" ;;
    *) die "用法: visevent krisspy|xche32 test|train|all" ;;
  esac

  local d="$DL/visevent"; mkdir -p "$d"

  if [ "$src" = "xche32" ]; then
    log "xche32/visevent 是 69GB 单包 tar.gz，内部结构为 visevent/{annos,annos.zip,check.py,test/<序列>/,train/<序列>/}"
    bash "$SELF" _fetch "$HF_HOST/datasets/xche32/visevent/resolve/main/visevent.tar.gz" "$d/visevent.tar.gz"
    local ext="$DATA/visevent"; mkdir -p "$ext"
    tar -xzf "$d/visevent.tar.gz" -C "$ext"
    # visevent/test -> visevent/train 挪到顶层并改名成代码要求的名字
    if [ -d "$ext/visevent" ]; then
      for pair in "test:testingset" "train:trainingset"; do
        ss="${pair%%:*}"; tt="${pair##*:}"
        if [ -d "$ext/visevent/$ss" ] && [ ! -d "$ext/$tt" ]; then
          mv "$ext/visevent/$ss" "$ext/$tt"; log "重命名 visevent/${ss} -> ${tt}"
        fi
      done
    fi
    _visevent_finish
    return 0
  fi

  [ "$src" = "krisspy" ] || die "未知源: $src"
  for s in $sets; do
    log "下载 VisEvent_$s （官方分包 zip，约 $([ "$s" = test ] && echo 102 || echo 131)GB）"
    # 先拿主 zip 的文件名清单
    curl -s --retry 5 --max-time 120 "$HF_HOST/api/datasets/krisspy39/visevent/tree/main/VisEvent_dataset?recursive=1" \
      | python3 -c "
import sys,json
want='$s'
for x in json.load(sys.stdin):
    p=x['path']
    if p.startswith('VisEvent_dataset/VisEvent_'+want) and (p.endswith('.zip') or '.z0' in p):
        print(p.split('/')[-1])
" > "$d/parts_$s.txt"
    [ -s "$d/parts_$s.txt" ] || die "拉取分包清单失败，检查能否访问 $HF_HOST"
    local plan="$d/plan_$s.txt"; : > "$plan"
    while read -r fn; do
      for host in "$HF_HOST" https://huggingface.co; do
        echo "$host/datasets/krisspy39/visevent/resolve/main/VisEvent_dataset/$fn $d/$fn" >> "$plan"
      done
    done < "$d/parts_$s.txt"
    xargs -P 1 -n 2 -r bash "$SELF" _fetch < "$plan"

    log "合并并解压 VisEvent_$s.zip"
    ( cd "$d"
      if command -v 7z >/dev/null 2>&1; then
        7z x -y "VisEvent_$s.zip" -o"$DATA/visevent"
      else
        command -v zip >/dev/null 2>&1 || die "需要 zip 或 7z： apt install zip"
        zip -F "VisEvent_$s.zip" --out "VisEvent_${s}_full.zip" -q
        unzip -q -o "VisEvent_${s}_full.zip" -d "$DATA/visevent"
      fi )
    log "  VisEvent_${s} 解压完成"
  done

  _visevent_finish
}

# VisEvent 收尾：把镜像的顶层目录名改成代码要求的名字 + 生成 testlist.txt
_visevent_finish() {
  for pair in "test_subset:testingset" "train_subset:trainingset"; do
    ss="${pair%%:*}"; tt="${pair##*:}"
    if [ -d "$DATA/visevent/$ss" ] && [ ! -d "$DATA/visevent/$tt" ]; then
      mv "$DATA/visevent/$ss" "$DATA/visevent/$tt"
      log "重命名 ${ss} -> ${tt}"
    fi
  done
  if [ -d "$DATA/visevent/testingset" ] && [ ! -f "$DATA/visevent/testingset/testlist.txt" ]; then
    warn "镜像里没有 testlist.txt，按目录自动生成（评测必需）"
    ( cd "$DATA/visevent/testingset" && ls -1 | grep -v '\.txt$' > testlist.txt )
  fi
  echo
  cmd_verify visevent
}

# ---------- 校验 ----------
_cmd_verify_depthtrack() {
  local t="$DATA/depthtrack/testingset" r="$DATA/depthtrack/trainingset"
  [ -d "$t" ] && echo "  DepthTrack test : $(ls -1 "$t" | wc -l) （应为 50）"  || echo "  DepthTrack test : 缺失"
  [ -d "$r" ] && echo "  DepthTrack train: $(ls -1 "$r" | wc -l) （应为 152；本仓训练清单只用 146）" || echo "  DepthTrack train: 未下载（训练时才需要）"
  if [ -d "$t" ] && [ "$(ls -1 "$t" | wc -l)" -gt 0 ]; then
    local first; first="$(ls -1 "$t" | head -1)"
    echo "  样例 $first 内容: $(ls "$t/$first" | tr '\n' ' ')"
    echo "  首帧: $(ls "$t/$first/color" 2>/dev/null | head -1) （必须 8 位补零从 1 开始）"
  fi
}
_cmd_verify_votrgbd() {
  local s="${SEQDIR:-$CODE/VOT22RGBD_workspace/sequences}"
  [ -d "$s" ] && echo "  VOT22-RGBD 序列: $(ls -1 "$s" | wc -l) （应为 127）" || echo "  VOT22-RGBD: 缺失"
  if [ -d "$s" ] && [ "$(ls -1 "$s" | wc -l)" -gt 0 ]; then
    local first; first="$(ls -1 "$s" | head -1)"
    echo "  样例 $first: $(ls "$s/$first" | tr '\n' ' ') | color 帧数 $(ls -1 "$s/$first/color" 2>/dev/null | wc -l)"
  fi
}
_cmd_verify_lasher() {
  local t="$DATA/lasher/testingset" r="$DATA/lasher/trainingset"
  [ -d "$t" ] && echo "  LasHeR test : $(ls -1 "$t" | wc -l) （应为 245）" || echo "  LasHeR test : 缺失"
  [ -d "$r" ] && echo "  LasHeR train: $(ls -1 "$r" | wc -l) （官方 979；本仓训练清单只用 881）" || echo "  LasHeR train: 未下载"
  if [ -d "$t" ] && [ "$(ls -1 "$t" | wc -l)" -gt 0 ]; then
    local first; first="$(ls -1 "$t" | head -1)"
    echo "  样例 $first 内容: $(ls "$t/$first" | tr '\n' ' ')"
    echo "  首帧名: $(ls "$t/$first/visible" | head -1) / $(ls "$t/$first/infrared" | head -1)  ← 两个模态必须同名且同宽度补零"
    echo "  GT 行数: $(wc -l < "$t/$first/visible.txt") / $(wc -l < "$t/$first/infrared.txt")"
  fi
}
_cmd_verify_visevent() {
  local r="$DATA/visevent/trainingset" t="$DATA/visevent/testingset"
  [ -d "$r" ] && echo "  VisEvent train: $(ls -1 "$r" | wc -l) （应为 500）" || echo "  VisEvent train: 缺失"
  [ -d "$t" ] && echo "  VisEvent test : $(ls -1 "$t" | wc -l) （应为 320）" || echo "  VisEvent test : 缺失"
  if [ -d "$t" ] && [ ! -f "$t/testlist.txt" ]; then
    warn "缺少 testlist.txt，正在生成（评测必需）"
    ( cd "$t" && ls -1 | grep -v '\.txt$' > testlist.txt )
  fi
  [ -f "$t/testlist.txt" ] && echo "  testlist.txt: $(wc -l < "$t/testlist.txt") 行"
  if [ -d "$t" ] && [ "$(ls -1 "$t" | wc -l)" -gt 0 ]; then
    local first; first="$(ls -1 "$t" | grep -v '\.txt$' | head -1)"
    [ -n "$first" ] || return 0
    echo "  样例 $first 内容: $(ls "$t/$first" | tr '\n' ' ')"
    echo "  vis_imgs $(ls -1 "$t/$first/vis_imgs" 2>/dev/null | wc -l) 帧 / event_imgs $(ls -1 "$t/$first/event_imgs" 2>/dev/null | wc -l) 帧 / GT $(wc -l < "$t/$first/groundtruth.txt") 行  ← 三者必须一致"
    echo "  首帧名: $(ls "$t/$first/vis_imgs" | head -1) （frame%04d.bmp，同宽度补零）"
  fi
}

cmd_verify() {
  log "校验结果："
  case "${1:-all}" in
    depthtrack) _cmd_verify_depthtrack ;;
    votrgbd)    _cmd_verify_votrgbd ;;
    lasher)     _cmd_verify_lasher ;;
    visevent)   _cmd_verify_visevent ;;
    all)        _cmd_verify_depthtrack; _cmd_verify_votrgbd; _cmd_verify_lasher; _cmd_verify_visevent ;;
    *) die "未知校验对象 $1" ;;
  esac
}

# ---------- 入口 ----------
main() {
  local cmd="${1:-help}"; shift || true
  case "$cmd" in
    check)      cmd_check "$@" ;;
    depthtrack) cmd_depthtrack "$@" ;;
    votrgbd)    cmd_votrgbd "$@" ;;
    lasher)     cmd_lasher "$@" ;;
    visevent)   cmd_visevent "$@" ;;
    verify)     cmd_verify "$@" ;;
    _fetch)     cmd__fetch "$@" ;;
    help|-h|*)  usage ;;
  esac
}
main "$@"
