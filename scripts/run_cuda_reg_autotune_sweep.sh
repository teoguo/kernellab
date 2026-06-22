#!/usr/bin/env bash
set -u

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

export PYTHONUSERBASE="${PYTHONUSERBASE:-/home/mnt/nas/c2smarter/python-userbase}"
export PATH="${PYTHONUSERBASE}/bin:${PATH}"

mkdir -p results/autotune
out="${AUTOTUNE_OUT:-results/autotune/sweep.csv}"
timed_iterations="${AUTOTUNE_ITERS:-5}"
echo "config,bm,bn,bk,tm,tn,threads,reg,smem,spill_stores,spill_loads,blocks_by_reg,blocks_by_smem,blocks_by_threads,theoretical_blocks,theoretical_occupancy_pct,verify128,verify_edge,verify512,gflops2048,pct2048,gflops4096,pct4096,status" > "$out"

if [ "${AUTOTUNE_FOCUS:-0}" = "2" ]; then
  configs=(
    "128 128 16 8 4" "128 128 16 8 8"
    "128 64 16 8 4" "128 64 16 8 8"
    "64 128 16 8 4" "64 128 16 8 8"
  )
elif [ "${AUTOTUNE_FOCUS:-0}" = "1" ]; then
  configs=(
    "128 128 8 8 4" "128 128 8 4 8" "128 128 8 8 8"
    "128 128 16 8 4" "128 128 16 4 8" "128 128 16 8 8"
    "128 64 8 8 4" "128 64 8 4 8" "128 64 8 8 8"
    "128 64 16 8 4" "128 64 16 4 8" "128 64 16 8 8"
    "64 128 8 8 4" "64 128 8 4 8" "64 128 8 8 8"
    "64 128 16 8 4" "64 128 16 4 8" "64 128 16 8 8"
  )
else
  configs=(
    "64 64 8 4 4" "64 64 8 8 4" "64 64 8 4 8"
    "64 64 16 4 4" "64 64 16 8 4" "64 64 16 4 8"
    "64 128 8 4 4" "64 128 8 8 4" "64 128 8 4 8" "64 128 8 8 8"
    "64 128 16 4 4" "64 128 16 8 4" "64 128 16 4 8" "64 128 16 8 8"
    "128 64 8 4 4" "128 64 8 8 4" "128 64 8 4 8" "128 64 8 8 8"
    "128 64 16 4 4" "128 64 16 8 4" "128 64 16 4 8" "128 64 16 8 8"
    "128 128 8 8 4" "128 128 8 4 8" "128 128 8 8 8"
    "128 128 16 8 4" "128 128 16 4 8" "128 128 16 8 8"
  )
fi

for cfg in "${configs[@]}"; do
  read -r bm bn bk tm tn <<< "$cfg"
  threads=$((bm * bn / (tm * tn)))
  name="BM${bm}_BN${bn}_BK${bk}_TM${tm}_TN${tn}"
  echo "=== ${name} threads=${threads} ==="

  build="build-cuda-autotune-sweep"
  flags="--ptxas-options=-v -DKLAB_REG_V2_BM=${bm} -DKLAB_REG_V2_BN=${bn} -DKLAB_REG_V2_BK=${bk} -DKLAB_REG_V2_TM=${tm} -DKLAB_REG_V2_TN=${tn}"
  if ! cmake -S . -B "$build" \
      -DCMAKE_BUILD_TYPE=Release \
      -DKERNELLAB_ENABLE_CUDA=ON \
      -DKERNELLAB_BUILD_TESTS=OFF \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc \
      -DCUDAToolkit_ROOT=/usr/local/cuda-12.1 \
      -DCMAKE_CUDA_FLAGS="$flags" >/dev/null; then
    echo "$name,$bm,$bn,$bk,$tm,$tn,$threads,,,,,,,,,,,,,,,,cmake_failed" >> "$out"
    continue
  fi

  log="results/autotune/${name}_ptxas.log"
  if ! cmake --build "$build" --target kernellab --clean-first -j 8 > "$log" 2>&1; then
    echo "$name,$bm,$bn,$bk,$tm,$tn,$threads,,,,,,,,,,,,,,,,build_failed" >> "$out"
    continue
  fi

  block="$(awk '/RegV2AutotunedKernel/{flag=1} flag{print} flag && /Used/{exit}' "$log")"
  reg="$(printf "%s\n" "$block" | sed -n 's/.*Used \([0-9][0-9]*\) registers.*/\1/p' | tail -1)"
  smem="$(printf "%s\n" "$block" | sed -n 's/.*registers, \([0-9][0-9]*\) bytes smem.*/\1/p' | tail -1)"
  spill_line="$(printf "%s\n" "$block" | grep 'spill stores' | tail -1 || true)"
  spill_stores="$(printf "%s\n" "$spill_line" | sed -n 's/.*0 bytes spill stores.*/0/p')"
  spill_loads="$(printf "%s\n" "$spill_line" | sed -n 's/.*0 bytes spill loads.*/0/p')"
  reg="${reg:-0}"
  smem="${smem:-0}"
  spill_stores="${spill_stores:-unknown}"
  spill_loads="${spill_loads:-unknown}"

  if [ "$reg" -gt 0 ]; then
    blocks_by_reg=$((65536 / (reg * threads)))
  else
    blocks_by_reg=0
  fi
  if [ "$smem" -gt 0 ]; then
    blocks_by_smem=$((102400 / smem))
  else
    blocks_by_smem=0
  fi
  blocks_by_threads=$((1536 / threads))
  blocks=$blocks_by_reg
  [ "$blocks_by_smem" -lt "$blocks" ] && blocks=$blocks_by_smem
  [ "$blocks_by_threads" -lt "$blocks" ] && blocks=$blocks_by_threads
  occ="$(awk -v b="$blocks" -v t="$threads" 'BEGIN { printf "%.1f", (b * t / 32.0 / 48.0) * 100.0 }')"

  v128=fail
  vedge=fail
  v512=fail
  if ./"$build"/kernellab verify --backend cuda_reg_v2 --m 128 --n 128 --k 128 >/dev/null; then
    v128=passed
  fi
  if ./"$build"/kernellab verify --backend cuda_reg_v2 --m 130 --n 129 --k 17 >/dev/null; then
    vedge=passed
  fi
  if ./"$build"/kernellab verify --backend cuda_reg_v2 --m 512 --n 512 --k 512 >/dev/null; then
    v512=passed
  fi
  if [ "$v128" != passed ] || [ "$vedge" != passed ] || [ "$v512" != passed ]; then
    echo "$name,$bm,$bn,$bk,$tm,$tn,$threads,$reg,$smem,$spill_stores,$spill_loads,$blocks_by_reg,$blocks_by_smem,$blocks_by_threads,$blocks,$occ,$v128,$vedge,$v512,,,,verify_failed" >> "$out"
    continue
  fi

  csv2048="results/autotune/${name}_2048.csv"
  csv4096="results/autotune/${name}_4096.csv"
  if ! ./"$build"/kernellab compare --backends cuda_reg_v2,cublas --m 2048 --n 2048 --k 2048 --warmup 3 --iterations "$timed_iterations" --csv-out "$csv2048" >/dev/null; then
    echo "$name,$bm,$bn,$bk,$tm,$tn,$threads,$reg,$smem,$spill_stores,$spill_loads,$blocks_by_reg,$blocks_by_smem,$blocks_by_threads,$blocks,$occ,$v128,$vedge,$v512,,,,bench2048_failed" >> "$out"
    continue
  fi
  if ! ./"$build"/kernellab compare --backends cuda_reg_v2,cublas --m 4096 --n 4096 --k 4096 --warmup 3 --iterations "$timed_iterations" --csv-out "$csv4096" >/dev/null; then
    echo "$name,$bm,$bn,$bk,$tm,$tn,$threads,$reg,$smem,$spill_stores,$spill_loads,$blocks_by_reg,$blocks_by_smem,$blocks_by_threads,$blocks,$occ,$v128,$vedge,$v512,,,,bench4096_failed" >> "$out"
    continue
  fi

  g2048="$(awk -F, '$6=="cuda_reg_v2"{print $16}' "$csv2048")"
  p2048="$(awk -F, '$6=="cuda_reg_v2"{print $17}' "$csv2048")"
  g4096="$(awk -F, '$6=="cuda_reg_v2"{print $16}' "$csv4096")"
  p4096="$(awk -F, '$6=="cuda_reg_v2"{print $17}' "$csv4096")"
  echo "$name,$bm,$bn,$bk,$tm,$tn,$threads,$reg,$smem,$spill_stores,$spill_loads,$blocks_by_reg,$blocks_by_smem,$blocks_by_threads,$blocks,$occ,$v128,$vedge,$v512,$g2048,$p2048,$g4096,$p4096,ok" >> "$out"
  tail -n 1 "$out"
done

printf "\nTop by 4096 GFLOPs:\n"
awk -F, 'NR>1 && $24=="ok" {print $22 "," $0}' "$out" | sort -t, -k1,1nr | head -8
