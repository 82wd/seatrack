#!/usr/bin/env bash
# Run on the user's Mac. Downloads only; does not execute remote commands.
set -uo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
result_dir="$project_dir/result"
remote="liziyan103@10.68.240.49"
key="$HOME/.ssh/id_ed25519_82wd"
remote_root="/nas/liziyan103/seatrack"
ok=0
skipped=0
failed=0
expected_paths=()

if [[ ! -f "$key" ]]; then
    echo "Missing SSH key: $key" >&2
    exit 1
fi

fetch() {
    local label="$1" source="$2" destination="$result_dir/$3"
    local parent stage name
    expected_paths+=("$destination")
    printf '\nDownloading: %s\n' "$label"
    if [[ -e "$destination" || -L "$destination" ]]; then
        printf 'SKIP (already exists, not revalidated): %s\n' "$destination"
        skipped=$((skipped + 1))
        return
    fi
    parent="$(dirname "$destination")"
    if ! mkdir -p "$parent"; then
        failed=$((failed + 1))
        return
    fi
    if ! stage="$(mktemp -d "$parent/.report-download.XXXXXX")"; then
        failed=$((failed + 1))
        return
    fi
    name="${source##*/}"
    # Transfer into a staging directory so a failed transfer is never marked complete.
    if scp -r -i "$key" -P 10358 "$remote:$source" "$stage/"; then
        if [[ -e "$stage/$name" && ! -e "$destination" && ! -L "$destination" ]] &&
           mv "$stage/$name" "$destination"; then
            rmdir "$stage"
            printf 'OK: %s\n' "$destination"
            ok=$((ok + 1))
        else
            printf 'FAILED to place download; retained at: %s\n' "$stage" >&2
            failed=$((failed + 1))
        fi
    else
        printf 'FAILED: %s\nPartial download (if any): %s\n' "$source" "$stage" >&2
        failed=$((failed + 1))
    fi
}

fetch 'RGBT234 self-trained metrics' \
    "$remote_root/SEATrack/RGBT_workspace/self_rgbt_ep0060_rgbt234/metrics_20261022_002510.json" \
    'rgbt234/2026-10-22_self-rgbt/reports/metrics_20261022_002510.json'

fetch 'VOT2022 self-trained report directory (JSON, HTML and assets)' \
    "$remote_root/evaluation_runs/self_vot22_ep0025/full/analysis/self_vot22_ep0025" \
    'vot22/2026-10-23_self-rgbd/reports'

fetch 'LasHeR official metrics' \
    "$remote_root/result/lasher/2026-10-11_official-rgbt/reports/lasher_metrics_python.json" \
    'lasher/2026-10-11_official-rgbt/reports/lasher_metrics_python.json'

fetch 'RGBT234 official metrics' \
    "$remote_root/evaluation_runs/20261013_remaining/rgbt234/official_rgbt234_full_20261013_074307_metrics.json" \
    'rgbt234/2026-10-13_official-rgbt/reports/official_rgbt234_full_20261013_074307_metrics.json'

fetch 'VisEvent self-trained metrics' \
    "$remote_root/SEATrack/RGBE_workspace/self_rgbe_ep0045/metrics_reference_20261014_141237_849005.json" \
    'visevent/2026-10-14_self-rgbe/reports/metrics_reference_20261014_141237_849005.json'

fetch 'DepthTrack self-trained report directory' \
    "$remote_root/SEATrack/SEATrack-main/Depthtrack_self_workspace/analysis/depthtrack_self" \
    'depthtrack/2026-10-11_self-rgbd/reports'

fetch 'RGBT234 checkpoint provenance' \
    "$remote_root/SEATrack/RGBT_workspace/self_rgbt_ep0060_rgbt234/checkpoint_used.json" \
    'rgbt234/2026-10-22_self-rgbt/provenance/checkpoint_used.json'

fetch 'VOT2022 checkpoint provenance' \
    "$remote_root/evaluation_runs/self_vot22_ep0025/checkpoint_used.json" \
    'vot22/2026-10-23_self-rgbd/provenance/checkpoint_used.json'

printf '\nFinished: downloaded=%s skipped=%s failed=%s\n' "$ok" "$skipped" "$failed"
printf 'Local result directory: %s\n' "$result_dir"
missing=0
for destination in "${expected_paths[@]}"; do
    if [[ ! -e "$destination" ]]; then
        printf 'MISSING after transfer: %s\n' "$destination" >&2
        missing=$((missing + 1))
    elif [[ -d "$destination" && ! -s "$destination/results.json" ]]; then
        printf 'MISSING/EMPTY JSON in report directory: %s\n' "$destination" >&2
        missing=$((missing + 1))
    fi
done
printf 'Final presence check: missing=%s\n' "$missing"
if ((missing > 0)); then
    exit 1
fi
if ((failed > 0)); then
    echo 'Some downloads failed. Send the error messages for path/connection checks.'
    exit 1
fi
