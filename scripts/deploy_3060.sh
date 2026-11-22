#!/usr/bin/env bash
# Setup, measurement, and one run on the 3060 box, in the order a run has to happen.
#
# This is a script rather than a list of commands because three of the steps can invalidate a run, and
# commands run one at a time are the kind of thing that drifts:
#   - the shape flags are the run key, so a pilot that uses different ones writes checkpoints the long run
#     refuses to resume;
#   - the corpus size is the corpus digest, the digest is inside the run key, and there is no append path,
#     so the size has to be chosen before the first document is written;
#   - TF32 is not inside the run key, so the directory is the only thing that keeps a TF32 run from
#     resuming into a checkpoint written by a run that did not use TF32.
#
# Everything the script can check, it checks before compute. What it cannot measure, it prints as not
# measured rather than as an estimate, and the two irreversible steps ask for a decision rather than
# assuming one.
#
# Run:
#   bash scripts/deploy_3060.sh
# Run it through bash rather than as ./scripts/deploy_3060.sh: the file was written on a filesystem with no
# executable bit, so git records it as 100644 and a Linux checkout has to be chmod'd or invoked through bash.
# `git update-index --chmod=+x scripts/deploy_3060.sh` is what makes it runnable directly.
#
# Every value below can be overridden from the environment. YES=1 is the non-interactive form of the
# confirmations, for a session where a prompt would hang.
set -euo pipefail

# ---------------------------------------------------------------------------
# Conditions. A record has to be able to attribute the numbers it contains, so every value that decides
# what a run means is stated here rather than read from a shell history.
# ---------------------------------------------------------------------------
REPO="${REPO:-$HOME/HeronLM}"
PYTHON="${PYTHON:-3.14}"
CUDA_INDEX="${CUDA_INDEX:-cu130}"

DATA_DIR="${DATA_DIR:-$HOME/data/fineweb-edu}"
CORPUS_DIR="${CORPUS_DIR:-$REPO/runs/corpus}"
RECORD_DIR="${RECORD_DIR:-$REPO/runs/records}"
PILOT_DIR="${PILOT_DIR:-$REPO/runs/pilot}"
CKPT_DIR="${CKPT_DIR:-$REPO/runs/phase0}"

# The run key is gate, name, cfg, seed, batch, seq_len, lr, wd, train_k, families, n_train, n_test and the
# corpus digest. `steps` is the only field a resume may change, which is what makes this script re-runnable:
# a second invocation with a larger STEPS extends the same run instead of starting a different one.
# lr, wd, train_k and families have no flags, so they come from the spec defaults; changing a default is a
# different run key, and a checkpoint written under the old default stops belonging to this run.
SEED="${SEED:-7}"
BATCH="${BATCH:-1}"
SEQ_LEN="${SEQ_LEN:-512}"
EVAL_WINDOWS="${EVAL_WINDOWS:-900}"
STEPS="${STEPS:-5000}"
CKPT_EVERY="${CKPT_EVERY:-1000}"
CKPT_KEEP="${CKPT_KEEP:-1}"
TF32="${TF32:-0}"

# Corpus size, stated as a condition. The fraction is a rule rather than a stop: `blake2b(seed:identity) <
# fraction` makes a corpus at 0.024 a strict subset of the same corpus at 1.0, and the same unit() decides
# the split, so the eval set is the same documents at every size. 0.024 is the fraction that lands near the
# 200M-token target if the source's own token claim is right; the cross-check the builder prints is the
# measurement that says whether it is, because that claim came from a different tokenizer.
TARGET_TOKENS="${TARGET_TOKENS:-200000000}"
SELECT_FRACTION="${SELECT_FRACTION:-0.024}"
VAL_FRACTION="${VAL_FRACTION:-0.02}"

# Raw crawl text contains tokens the tokenizer marks as special, and a document whose boundary token is one of
# them is a measurement problem rather than a representation problem. The default is `skip`, and the count of
# what was skipped is in the manifest, so a run can say what its eval set is missing.
SPECIAL_POLICY="${SPECIAL_POLICY:-skip}"

# The pilot evaluates few windows on purpose. The eval pass is a fixed cost per gate, not a per-step cost, so
# a pilot that evaluates the run's 900 windows cannot separate the two, and the per-step number it reports
# would be the wrong shape for the projection. The eval cost stays a stated unknown.
PILOT_STEPS="${PILOT_STEPS:-20}"
PILOT_EVAL_WINDOWS="${PILOT_EVAL_WINDOWS:-8}"

# The pilot runs twice by default, once at each arithmetic. There is no autocast anywhere in the harness, so
# on Ampere the card's fp32 speed is almost entirely TF32, and the only way to know whether the gain is real
# is to measure the same 20 steps both ways. The second pilot is a throughput measurement: its gate numbers
# are a different arithmetic and cannot be compared with the first.
PILOT_TF32_COMPARE="${PILOT_TF32_COMPARE:-1}"

# Two limits the script can enforce, both from measurements rather than from a rule of thumb.
MAX_VRAM_MIB="${MAX_VRAM_MIB:-11264}"
MAX_HOURS="${MAX_HOURS:-12}"

PY="$REPO/.venv/bin/python"
problems=()

problem() {
    # Refusals are collected rather than raised, so a run reports every reason it would be invalid rather
    # than the first one.
    problems+=("$1")
}

fits_12gb() {
    # Measured, not guessed, fp32 with no autocast anywhere in the harness: batch 1 at 512 peaks about 8.3 GB,
    # batch 1 at 1024 about 10.5 GB with about 1 GB of margin. Batch 2 at 1024 and batch 1 at 2048 do not fit,
    # so they are refused here rather than attempted by a run that spends a day to find out.
    case "${1}x${2}" in
        1x512|1x1024) return 0 ;;
        *) return 1 ;;
    esac
}

confirm() {
    if [ "${YES:-0}" = "1" ]; then
        return 0
    fi
    if [ ! -t 0 ]; then
        problem "${1}: not interactive and YES is not set"
        return 1
    fi
    read -r -p "confirm: ${1} [y/N] " answer
    case "$answer" in
        [yY]|[yY][eE][sS]) return 0 ;;
        *)
            problem "${1}: not confirmed"
            return 1
            ;;
    esac
}

report_problems() {
    if [ "${#problems[@]}" -eq 0 ]; then
        return 0
    fi
    printf '\nrefused:\n'
    for entry in "${problems[@]}"; do
        printf '  %s\n' "$entry"
    done
    exit 2
}

# ---------------------------------------------------------------------------
# 1. Guards, before anything is downloaded, installed or run.
# ---------------------------------------------------------------------------
printf '\n1. guards\n'

# The checks in this script are GNU-coreutils checks: `df --output=avail`, `sha256sum` and the driver-version
# split all read differently on a BSD df or a macOS sha256sum. Rather than produce a number that would be
# wrong on another platform, the script says which platform it was written for.
os=$(uname -s)
if [ "$os" = "Linux" ]; then
    :
else
    problem "this script was written for Linux; uname reports ${os}, so the disk and pin checks below are GNU-coreutils calls that read differently on another platform"
fi

command -v uv >/dev/null 2>&1 || problem "uv is not installed; the install step below uses it"

sampler_available=0
if command -v nvidia-smi >/dev/null 2>&1; then
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n 1)
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
    sampler_available=1
    printf '  card      : %s, %s MiB, driver %s\n' "$gpu_name" "$gpu_mib" "$driver"
    # A number that came from a query is only compared when it is a number. An empty or non-numeric field is
    # reported as not measured rather than compared as though it were a reading.
    case "$gpu_mib" in
        '' | *[!0-9]*)
            problem "the card reports ${gpu_mib} as its memory, which is not a number, so the measured shapes below cannot be checked against it"
            ;;
        *)
            if [ "$gpu_mib" -lt 12288 ]; then
                problem "this card reports ${gpu_mib} MiB, so the measured shapes below are not this card's numbers"
            fi
            ;;
    esac
    # The cu130 wheels need a driver in the R580 branch. An older driver can use cu126, but the record then
    # describes a different CUDA build, and a number is only comparable with a number measured under it. Only
    # the leading branch is a number: a full version is 580.65.04, and the comparison is on the 580.
    branch=${driver%%.*}
    case "$branch" in
        '' | *[!0-9]*)
            problem "the driver reports ${driver} as its version, so the branch is not a number and the R580 branch cu130 needs cannot be checked"
            ;;
        *)
            if [ "$CUDA_INDEX" = "cu130" ] && [ "$branch" -lt 580 ]; then
                problem "driver ${driver} is below the R580 branch cu130 needs; run with CUDA_INDEX=cu126 and accept that this run is comparable only with another cu126 run"
            fi
            ;;
    esac
else
    problem "nvidia-smi is not available, so the card cannot be checked before compute and its memory cannot be sampled during the pilot. torch below is still the check that the card is usable, but the peak-memory part of this script stays unmeasured."
fi

# Precision is not in the run key, so nothing but the directory stops a TF32 run from resuming into a
# checkpoint written without it. Keeping them apart is the only check the harness can make.
if [ "$TF32" = "1" ]; then
    CKPT_DIR="$REPO/runs/phase0_tf32"
    printf '  tf32      : on. These numbers are only comparable with another TF32 run, so they live in a separate checkpoint directory (%s).\n' "$CKPT_DIR"
fi

if ! fits_12gb "$BATCH" "$SEQ_LEN"; then
    problem "batch ${BATCH} seq ${SEQ_LEN} is not a shape that was measured to fit 12 GB; use 1x512 or 1x1024"
fi

# Storage, stated as what it is made of: data about 30 GB, token shards about 1 GB, and the checkpoints. A
# record is params plus buffers plus two Adam moments, which at PHASE0 is 5.13 GB per unique training run, and
# the ten arm entries share six unique runs. `--checkpoint-keep 1` is the thing that bounds it: an unbounded
# interval at the ROADMAP budget is a few hundred records per run rather than six.
need_gib=64
free_gib=""
if command -v df >/dev/null 2>&1; then
    free_gib=$(df -k --output=avail "$HOME" 2>/dev/null | tail -n 1 | awk 'NF { printf "%d", $1 / 1048576 }') || free_gib=""
fi
if [ -z "$free_gib" ]; then
    problem "free space at ${HOME} is not measured, so the ${need_gib} GiB the corpus and the checkpoints need cannot be checked against it"
else
    printf '  disk      : %s GiB free, %s GiB needed (data, shards, %s record per run at 5.13 GB)\n' \
        "$free_gib" "$need_gib" "$CKPT_KEEP"
    if [ "$free_gib" -lt "$need_gib" ]; then
        problem "${free_gib} GiB free is below the ${need_gib} GiB the corpus and the checkpoints need"
    fi
fi

report_problems

# ---------------------------------------------------------------------------
# 2. Environment. Idempotent, so a second invocation is a resume rather than a rebuild.
# ---------------------------------------------------------------------------
printf '\n2. environment\n'

[ -d "$REPO/.venv" ] || (cd "$REPO" && uv venv --python "$PYTHON" .venv)
cd "$REPO"

# torch comes from the CUDA index. PyPI's Linux wheel for torch is the CPU build, and `torch>=2.6` in
# pyproject.toml is satisfied by it, so a plain install quietly installs the wrong thing. --no-deps keeps the
# resolver from replacing the CUDA wheel when the project itself is installed.
uv pip install --index-url "https://download.pytorch.org/whl/${CUDA_INDEX}" torch
uv pip install --no-deps -e .
uv pip install gguf numpy
# Builder-only, deliberately not declared in pyproject.toml: the reference does not read parquet or the
# tokenizer file, only the CLI does.
uv pip install tokenizers pyarrow huggingface_hub

"$PY" -c 'import sys, torch
if sys.version_info[:2] < (3, 14):
    raise SystemExit("venv python is " + sys.version.split()[0] + "; pyproject.toml requires >=3.14, and the record does not state the interpreter, so a run here is not comparable with the CI run")
print("  python    :", sys.version.split()[0])
print("  torch     :", torch.__version__, "cuda build", torch.version.cuda, "available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("torch cannot see the card: the run would train on CPU and the record would say so, but a day of CPU is not the measurement this script is for")
print("  card      :", torch.cuda.get_device_name(0))'

# The suites are the check that this checkout is the same checkout. Line endings are the risk on a different
# machine, and the pinned tokenizer's identity is a digest of its bytes: a checkout that turned tokenizer.json
# into CRLF is a different vocabulary, not the same vocabulary with different line endings. The digest the
# corpus manifest stores is the one this run is attributed to, so it is printed here.
if command -v sha256sum >/dev/null 2>&1; then
    printf '  pin       : %s\n' "$(sha256sum tokenizer/qwen2.5/tokenizer.json | cut -d' ' -f1)"
else
    printf '  pin       : not printed. sha256sum is not available, so the bytes are not identified here, and the suite below is the only check that can say whether this checkout is the same checkout.\n'
fi
"$PY" tests/run_tests.py

# ---------------------------------------------------------------------------
# 3. Corpus, once. There is no append path, so this is the only moment the size can be chosen.
# ---------------------------------------------------------------------------
printf '\n3. corpus\n'

if [ -f "$CORPUS_DIR/manifest.json" ]; then
    printf '  exists    : not rebuilt. A rebuild would change the corpus digest, which is inside the run key, so every record written against the current corpus would stop belonging to this run.\n'
else
    val_windows=$(awk -v t="$TARGET_TOKENS" -v f="$VAL_FRACTION" -v s="$SEQ_LEN" 'BEGIN { printf "%d", t * f / s }')
    printf '  size      : %s of the source by identity hash, against a target of %s tokens. The fraction is chosen now and cannot be changed later; the cross-check the builder prints is the measurement that says whether the fraction was right, because the source states its counts in its own tokenizer.\n' \
        "$SELECT_FRACTION" "$TARGET_TOKENS"
    printf '  eval set  : val fraction %s, whole documents, which is about %s windows at seq_len %s. ROADMAP states 4,000,000 tokens as its eval condition, and %s windows is above the 900-window error-bar floor. Both settings are defensible and only one can be chosen, so this is the last point at which it can be.\n' \
        "$VAL_FRACTION" "$val_windows" "$SEQ_LEN" "$val_windows"

    if confirm "download about 30 GB and build the corpus at this size"; then
        if command -v hf >/dev/null 2>&1; then
            hf_tool="hf"
        elif command -v huggingface-cli >/dev/null 2>&1; then
            hf_tool="huggingface-cli"
        else
            problem "neither hf nor huggingface-cli is installed, so the source cannot be downloaded"
            report_problems
        fi

        # Row identity is positional: (file, row_index). FineWeb-Edu has no id column, so nothing makes an
        # identity stable across a re-sort, and a partial download is a different document set, which is a
        # different corpus digest. This is why the data directory is kept after the build rather than cleaned
        # up: the corpus is only reproducible from the same files.
        if [ -z "$(ls -A "$DATA_DIR" 2>/dev/null)" ]; then
            "$hf_tool" download --repo-type dataset HuggingFaceFW/fineweb-edu --include 'sample/10BT/*' \
                --local-dir "$DATA_DIR"
        else
            printf '  data      : already present, not re-downloaded\n'
        fi
        found=$(find "$DATA_DIR" -name '*.parquet' 2>/dev/null | wc -l) || found=0
        printf '  files     : %s parquet files. sample/10BT publishes 14, so a different count is a different document set and therefore a different corpus id, which is the number the run key contains.\n' "$found"
        if [ "$found" -ne 14 ] && ! confirm "build from ${found} files rather than the 14 the source publishes"; then
            report_problems
        fi

        "$PY" scripts/tokenize_corpus.py --tokenizer tokenizer/qwen2.5/tokenizer.json \
            --parquet-dir "$DATA_DIR" --out "$CORPUS_DIR" \
            --select-fraction "$SELECT_FRACTION" --val-fraction "$VAL_FRACTION" \
            --seed "$SEED" --special-policy "$SPECIAL_POLICY"
    fi
fi

report_problems

# ---------------------------------------------------------------------------
# 4. Measurement. The pilot writes no checkpoints, so it cannot poison the run: its eval size is already a
# different run key, and the only way a pilot could invalidate the long run is by writing a checkpoint under a
# key the long run would then refuse. Keeping it out of the checkpoint directory is what makes that harmless.
# ---------------------------------------------------------------------------
printf '\n4. measurement\n'

# On Linux preflight prints the arithmetic from the module shapes and its own measured time, and prints no
# peak-memory line, because `process_peak_bytes()` is a Windows psapi call. The card's memory is sampled below
# instead. Its checkpoint figure counts params twice - once as params, once inside the optimizer state - so
# read that line as params plus two Adam moments.
"$PY" tests/preflight.py

sample_file=""
sampler_pid=""

stop_sampler() {
    if [ -n "$sampler_pid" ]; then
        kill "$sampler_pid" 2>/dev/null || true
        wait "$sampler_pid" 2>/dev/null || true
    fi
}

# The sampler is the only thing this script leaves running, so the exit handler closes it on every path,
# including a path that exits before the measurement section reaches its own stop. A sampler that outlives
# the script keeps writing samples attributed to a run that never happened.
trap 'stop_sampler' EXIT

if [ "$sampler_available" = "1" ]; then
    mkdir -p "$REPO/runs"
    sample_file="$REPO/runs/vram_samples.txt"
    : > "$sample_file"
    # memory.used is a sample, not a peak, so the true peak is at least the highest line this writes. The
    # distinction matters because the 12 GB limit is a limit on the peak, not on the sample.
    while :; do
        nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> "$sample_file" 2>/dev/null || break
        sleep 2
    done &
    sampler_pid=$!
fi

declare -A wall

pilot() {
    # $1 is the record-directory label, $2 is the TF32 flag or an empty string. Wall time is stored per label
    # so the two pilots can be compared; the driver's own per-step number is not used here because the driver
    # prints the eval pass inside the same run and the two cannot be separated by the record.
    local label="$1" flag="$2" start code
    start=$SECONDS
    code=0
    "$PY" scripts/phase0.py --scale phase0 --steps "$PILOT_STEPS" --batch "$BATCH" --seq-len "$SEQ_LEN" \
        --seed "$SEED" --eval-windows "$PILOT_EVAL_WINDOWS" --corpus-dir "$CORPUS_DIR" \
        --record-dir "$PILOT_DIR/$label" $flag || code=$?
    wall[$label]=$((SECONDS - start))
    return "$code"
}

rate() {
    awk -v t="${wall[$1]}" -v n="$PILOT_STEPS" 'BEGIN { printf "%.1f", t / n }'
}

flag_a=""
if [ "$TF32" = "1" ]; then
    flag_a="--tf32"
fi
code_a=0
pilot configured "$flag_a" || code_a=$?
if [ "$code_a" -ne 0 ]; then
    # A refused measurement is a problem rather than a message, because the projection and the run
    # confirmation both read from it. The exit handler closes the sampler on this path.
    problem "pilot a exited ${code_a}, so the per-step rate is not measured and the run cannot be confirmed against a projection that does not exist"
    report_problems
fi

rate_a=$(rate configured)
printf '  pilot a   : %s s per training step (%s s wall over %s steps), at the arithmetic this run is configured for (%s).\n' \
    "$rate_a" "${wall[configured]}" "$PILOT_STEPS" "$( [ "$TF32" = "1" ] && echo 'TF32 on' || echo 'TF32 off' )"

if [ "$PILOT_TF32_COMPARE" = "1" ] && [ "$sampler_available" = "1" ]; then
    # The other setting is the one the run was not configured for, and `name_b` states which one it was.
    flag_b="--tf32"
    name_b="TF32 on"
    if [ "$TF32" = "1" ]; then
        flag_b=""
        name_b="TF32 off"
    fi
    code_b=0
    pilot other "$flag_b" || code_b=$?
    if [ "$code_b" -ne 0 ]; then
        printf '  pilot b   : exited %s, so the comparison is not measured.\n' "$code_b"
    else
        rate_b=$(rate other)
        ratio=$(awk -v a="$rate_a" -v b="$rate_b" 'BEGIN { if (b > 0) printf "%.2f", a / b; else print "not computable" }')
        printf '  pilot b   : %s s per training step, the same 20 steps at %s. Both pilots share a run key, because TF32 is not in the key, which is why they wrote different record directories: the record is the only place the difference between them is visible.\n' \
            "$rate_b" "$name_b"
        printf '  difference  : %s. About 1.00 means the card gains nothing from TF32 and the run can stay on the arithmetic the CPU reference produces; a large ratio means the run is faster, but every number it writes is comparable only with another run at the same setting.\n' "$ratio"
    fi
fi

stop_sampler

projected="not projected"
if [ "${wall[configured]}" -gt 0 ]; then
    hours=$(awk -v s="$rate_a" -v n="$STEPS" 'BEGIN { printf "%.1f", s * n / 3600 }')
    projected="$hours"
    printf '  projected : %s steps of training is %s h at %s s per step, with no card idle time and no eval pass in the number.\n' \
        "$STEPS" "$hours" "$rate_a"
    printf '  eval      : not measured by this pilot. The eval pass is a fixed cost per gate rather than a per-step cost, so at %s windows per arm over k=1..4 it is the larger part of a short run and the wall clock is not %s h.\n' \
        "$EVAL_WINDOWS" "$hours"
    printf '  gates     : at %s steps nothing numeric is decidable. Only the pattern thresholds decide at any scale, so a short run reports indicative rather than pass or fail.\n' "$STEPS"
    if awk -v h="$hours" -v m="$MAX_HOURS" 'BEGIN { exit !(h > m) }'; then
        if ! confirm "start a run projected at ${hours} h, above the ${MAX_HOURS} h this script was written for"; then
            report_problems
        fi
    fi
else
    printf '  projected : not measured, because the pilot recorded no wall time\n'
fi

if [ -n "$sample_file" ] && [ -s "$sample_file" ]; then
    peak_mib=$(awk 'BEGIN { m = 0 } { if ($1 > m) m = $1 } END { print m }' "$sample_file")
    printf '  vram      : %s MiB was the highest sample during the pilot (a sample, so the peak is at least this)\n' "$peak_mib"
    if [ "$peak_mib" -gt "$MAX_VRAM_MIB" ]; then
        problem "the pilot reached ${peak_mib} MiB at batch ${BATCH} seq ${SEQ_LEN}, above the ${MAX_VRAM_MIB} MiB limit for a 12 GB card; lower BATCH or SEQ_LEN before the long run"
    fi
else
    printf '  vram      : not measured\n'
fi

report_problems

# ---------------------------------------------------------------------------
# 5. The run. The driver refuses an eval set larger than the val split's full windows and an --eval-tokens that
# is not a whole number of windows, so those two are not duplicated here. --resume is left on because a missing
# checkpoint starts the run rather than failing it, which is what makes a second invocation of this script an
# extension of the same run.
# ---------------------------------------------------------------------------
printf '\n5. run\n'
printf '  run key   : seed=%s batch=%s seq_len=%s eval_windows=%s corpus=%s, plus lr, wd, train_k and families from the spec defaults\n' \
    "$SEED" "$BATCH" "$SEQ_LEN" "$EVAL_WINDOWS" "$CORPUS_DIR"
printf '  records   : %s, every %s steps, keep %s. Records are counted per unique training run, not per arm label.\n' \
    "$CKPT_DIR" "$CKPT_EVERY" "$CKPT_KEEP"

if ! confirm "run ${STEPS} steps, projected at ${projected} h"; then
    report_problems
fi

run_args=(--scale phase0 --steps "$STEPS" --batch "$BATCH" --seq-len "$SEQ_LEN" --seed "$SEED"
    --eval-windows "$EVAL_WINDOWS" --corpus-dir "$CORPUS_DIR"
    --checkpoint-dir "$CKPT_DIR" --checkpoint-every "$CKPT_EVERY" --checkpoint-keep "$CKPT_KEEP"
    --record-dir "$RECORD_DIR" --resume)
if [ "$TF32" = "1" ]; then
    run_args+=(--tf32)
fi

code=0
"$PY" scripts/phase0.py "${run_args[@]}" || code=$?
printf 'exit=%s\n' "$code"
exit "$code"
