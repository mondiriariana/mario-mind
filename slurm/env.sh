# Sourced by every Slurm script (jobs are submitted from the code folder on the cluster).
# Site paths come from slurm/paths.local.sh, which is not in git: copy slurm/paths.example.sh.
_MJG_SLURM_DIR="${SLURM_SUBMIT_DIR:-$PWD}/slurm"
if [ -f "$_MJG_SLURM_DIR/paths.local.sh" ]; then
  source "$_MJG_SLURM_DIR/paths.local.sh"
fi
: "${MJG_ROOT:?set MJG_ROOT in slurm/paths.local.sh}"
export MJG_CODE=${MJG_CODE:-$MJG_ROOT/code}
export MJG_SCRATCH=${MJG_SCRATCH:-$MJG_ROOT/scratch}
export MJG_MODELS=${MJG_MODELS:-$MJG_ROOT/models}
export MJG_TMP=$MJG_SCRATCH/tmp/${SLURM_JOB_ID:-local}  # per-job temp: replayed frames, vLLM sockets
export TMPDIR=$MJG_TMP
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
mkdir -p "$MJG_CODE/logs" "$MJG_TMP"
cd "$MJG_CODE"
