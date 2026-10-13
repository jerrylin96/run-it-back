#!/bin/bash
# ============================================================================
# Confirm the CPU partition has outbound internet to GCS BEFORE committing the ~16 TB
# conversion to it. The conversion is CPU/network-only, so it should run off the GPUs,
# but only if the partition can actually reach GCS. Run this first:
#   cd <repo root>; sbatch data_process/net_probe.sbatch ; then read the .log
# PASS  -> run convert_era5.sbatch as-is (targets the cpu partition).
# FAIL  -> run convert_era5.sbatch on a partition with outbound internet.
# ============================================================================
#SBATCH --job-name=makani-netprobe
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=makani-netprobe-%j.log
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE,INVALID_DEPEND,TIME_LIMIT

set -uo pipefail   # NOT -e: we want to capture a curl failure and report it, not abort

# Probe the EXACT bucket convert_era5.sbatch downloads from (canonical ARCO). An anonymous 200
# here confirms BOTH that the cpu partition has outbound internet AND that the ARCO bucket allows
# anonymous (non-requester-pays) reads — the two things the 18 TB conversion depends on.
URL="https://storage.googleapis.com/gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3/.zmetadata"
echo "Probing outbound HTTPS to the canonical ARCO bucket from $(hostname) (partition: cpu) ..."
code=$(curl -s -o /dev/null -w '%{http_code}' --max-time 60 "$URL" || echo "000")
echo "GET $URL"
echo "HTTP status: $code"

if [ "$code" = "200" ]; then
  echo "RESULT: PASS — this partition reaches the ARCO bucket anonymously. Run convert_era5.sbatch as-is."
else
  echo "RESULT: FAIL (status $code). Two possible causes, two fallbacks:"
  echo "  - no internet on this partition -> submit convert_era5.sbatch to one that has it;"
  echo "  - ARCO bucket needs a billing project (requester-pays) -> use the WB2 store + 70-ch fallback"
  echo "    (ZARR=gs://weatherbench2/.../...-chunk-1.zarr-v2, META=metadata_fcn3_70.json)."
fi
