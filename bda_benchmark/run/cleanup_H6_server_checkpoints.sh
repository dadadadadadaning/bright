#!/usr/bin/env bash
set -euo pipefail

# Safe cleanup for the active H6-SDCR route.
#
# Default: preview only.
# Delete:  bash run/cleanup_H6_server_checkpoints.sh --execute
# The execute mode additionally requires typing the confirmation phrase.

MODE="${1:---preview}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-/root/shared-nvme/BRIGHT-checkpoints}"
EXPECTED_ROOT="/root/shared-nvme/BRIGHT-checkpoints"

case "${MODE}" in
  --preview) ;;
  --execute) ;;
  *)
    echo "Usage: $0 [--preview|--execute]" >&2
    exit 2
    ;;
esac

resolved_root="$(realpath -m -- "${CHECKPOINT_ROOT}")"
if [[ "${resolved_root}" != "${EXPECTED_ROOT}" ]]; then
  echo "Refusing cleanup: unexpected checkpoint root: ${resolved_root}" >&2
  exit 1
fi
if [[ ! -d "${resolved_root}" ]]; then
  echo "Checkpoint root does not exist: ${resolved_root}" >&2
  exit 1
fi

# These four inputs are the immutable dependency set of H6 A1/A2/A3.
required_files=(
  "${resolved_root}/S8_MICB_DSMIL/BRIGHT/S8_P3_DSMIL_EBS16/best_val_miou.pth"
  "${resolved_root}/H5_BIQ/source/H4_A4_step5500_valid602784_locked.pth"
  "${resolved_root}/HeterogeneousDamageFormer/sar_train_stats.json"
  "${resolved_root}/H4_RGBSAR/damage_train_priors.json"
)

echo "Checking protected H6 dependencies..."
for path in "${required_files[@]}"; do
  if [[ ! -s "${path}" ]]; then
    echo "Refusing cleanup: protected dependency is missing: ${path}" >&2
    exit 1
  fi
  echo "PROTECTED  ${path}"
done

# Explicit deletion allowlist.  It intentionally excludes:
#   H6_A1_pilot3000_seed3407
#   H6_A2_pilot3000_seed3407
#   S8_MICB_DSMIL
#   H5_BIQ (because its source/ directory is active)
#   H4_RGBSAR and HeterogeneousDamageFormer (because each contains one active JSON)
targets=(
  "${resolved_root}/H6_SDCR/H6_A0_guard_seed3407"
  "${resolved_root}/H6_SDCR/H6_A1_anchor_guard_seed3407"
  "${resolved_root}/H6_SDCR/H6_A1_smoke20_seed3407"
  "${resolved_root}/H6_SDCR/H6_A2_smoke20_seed3407"
  "${resolved_root}/H6_SDCR/H6_A3_smoke20_seed3407"
  "${resolved_root}/H5_BC"
  "${resolved_root}/S8_P5_C2_Recovery"
  "${resolved_root}/S8_P5_DCMF"
  "${resolved_root}/S8_P4_NSMICB"
  "${resolved_root}/S8"
  "${resolved_root}/S9_XHPI"
  "${resolved_root}/S3V2"
)

validate_target() {
  local target="$1"
  local resolved
  resolved="$(realpath -m -- "${target}")"
  case "${resolved}" in
    "${resolved_root}"/*) ;;
    *)
      echo "Unsafe target outside checkpoint root: ${resolved}" >&2
      return 1
      ;;
  esac
  if [[ "${resolved}" == "${resolved_root}" ]]; then
    echo "Refusing to delete checkpoint root" >&2
    return 1
  fi
  for protected in "${required_files[@]}"; do
    case "${protected}" in
      "${resolved}"|"${resolved}"/*)
        echo "Target would contain protected dependency: ${resolved}" >&2
        return 1
        ;;
    esac
  done
}

echo
echo "Filesystem before cleanup:"
df -h "${resolved_root}"
echo
echo "Deletion plan:"
existing=()
for target in "${targets[@]}"; do
  validate_target "${target}"
  if [[ -e "${target}" ]]; then
    size="$(du -sh -- "${target}" | awk '{print $1}')"
    printf 'DELETE  %-8s %s\n' "${size}" "${target}"
    existing+=("${target}")
  else
    printf 'SKIP    %-8s %s\n' "missing" "${target}"
  fi
done

echo
echo "Explicitly retained H6 results:"
for keep in \
  "${resolved_root}/H6_SDCR/H6_A1_pilot3000_seed3407" \
  "${resolved_root}/H6_SDCR/H6_A2_pilot3000_seed3407"; do
  if [[ -d "${keep}" ]]; then
    du -sh -- "${keep}"
  else
    echo "WARNING: expected Pilot directory is missing: ${keep}"
  fi
done

if [[ "${MODE}" == "--preview" ]]; then
  echo
  echo "Preview only; nothing was deleted."
  echo "After confirming that these old routes were downloaded to your PC, run:"
  echo "  bash run/cleanup_H6_server_checkpoints.sh --execute"
  exit 0
fi

if [[ "${#existing[@]}" -eq 0 ]]; then
  echo "Nothing to delete."
  exit 0
fi

confirmation="${CONFIRM_H6_CLEANUP:-}"
if [[ -z "${confirmation}" ]]; then
  read -r -p "Type DELETE_ARCHIVED_H6_FILES to continue: " confirmation
fi
if [[ "${confirmation}" != "DELETE_ARCHIVED_H6_FILES" ]]; then
  echo "Confirmation did not match; nothing was deleted." >&2
  exit 1
fi

stamp="$(date +%Y%m%d_%H%M%S)"
manifest_dir="/root/shared-nvme/BRIGHT-train-logs/H6_SDCR"
mkdir -p "${manifest_dir}"
manifest="${manifest_dir}/cleanup_${stamp}.tsv"
printf 'timestamp\tsize_before\tpath\n' > "${manifest}"

for target in "${existing[@]}"; do
  validate_target "${target}"
  size="$(du -sh -- "${target}" | awk '{print $1}')"
  printf '%s\t%s\t%s\n' "$(date '+%F %T')" "${size}" "${target}" >> "${manifest}"
  echo "Deleting verified target: ${target} (${size})"
  rm -rf --one-file-system -- "${target}"
done

echo
echo "Rechecking protected dependencies..."
for path in "${required_files[@]}"; do
  test -s "${path}" || {
    echo "CRITICAL: protected dependency disappeared: ${path}" >&2
    exit 1
  }
  echo "OK  ${path}"
done

echo
echo "Filesystem after cleanup:"
df -h "${resolved_root}"
echo "Cleanup manifest: ${manifest}"
echo "Cleanup completed."
