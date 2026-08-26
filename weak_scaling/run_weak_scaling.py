#!/usr/bin/env python3
"""
Weak-scaling experiment generator/runner for iPIC3D-GPU on LUMI-G.

Baseline: 1 node / 8 GPUs. Each additional step adds one more node (8 more
GPUs) and grows the domain so the amount of work per GPU (cells/GPU) stays
constant. For each node count this writes a self-contained run directory
(input file + slurm script) under weak_scaling/N<k>/, derived from a base
.inp template and a base slurm template.

Usage:
  ./run_weak_scaling.py                       # generate N=1..4, do not submit
  ./run_weak_scaling.py --max-nodes 8         # generate N=1..8
  ./run_weak_scaling.py --max-nodes 8 --submit  # generate and sbatch each
  ./run_weak_scaling.py --collect             # summarize results from logs
"""
import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_INPUT = REPO_ROOT / "testGEM3D.inp"
DEFAULT_BASE_SLURM = REPO_ROOT / "runIMM.slurm"
DEFAULT_EXECUTABLE = REPO_ROOT / "build" / "iPIC3D"
OUT_ROOT = Path(__file__).resolve().parent

DIMS = ("x", "y", "z")
KEY_OF = {"x": ("Lx", "nxc", "XLEN"), "y": ("Ly", "nyc", "YLEN"), "z": ("Lz", "nzc", "ZLEN")}


def balanced_factors(n: int):
    """Return (x, y, z) with x*y*z == n, minimizing max(x,y,z)/min(x,y,z)."""
    best = None
    for x in range(1, n + 1):
        if n % x:
            continue
        rem = n // x
        for y in range(1, rem + 1):
            if rem % y:
                continue
            z = rem // y
            spread = max(x, y, z) / min(x, y, z)
            if best is None or spread < best[0]:
                best = (spread, (x, y, z))
    return best[1]


def read_kv(text: str, key: str) -> float:
    m = re.search(rf"^\s*{key}\s*=\s*(\S+)", text, re.MULTILINE)
    if not m:
        raise ValueError(f"key {key!r} not found in base input")
    return float(m.group(1))


def set_kv(text: str, key: str, value) -> str:
    pattern = rf"^(\s*{key}\s*=\s*)\S+"
    if isinstance(value, float):
        repl = f"\\g<1>{value:.6f}"
    else:
        repl = f"\\g<1>{value}"
    new_text, n = re.subn(pattern, repl, text, count=1, flags=re.MULTILINE)
    if n == 0:
        raise ValueError(f"key {key!r} not found in base input")
    return new_text


def make_input(base_text: str, nodes: int, gpus_per_node: int, per_gpu_cells: int, sim_name: str) -> str:
    total_ranks = nodes * gpus_per_node
    factors = sorted(balanced_factors(total_ranks), reverse=True)

    base_n = {d: read_kv(base_text, KEY_OF[d][1]) for d in DIMS}
    base_L = {d: read_kv(base_text, KEY_OF[d][0]) for d in DIMS}
    order = sorted(DIMS, key=lambda d: base_n[d], reverse=True)
    flen = dict(zip(order, factors))

    text = base_text
    for d in DIMS:
        Lkey, nkey, xkey = KEY_OF[d]
        new_n = int(flen[d] * per_gpu_cells)
        new_L = (base_L[d] / base_n[d]) * new_n
        text = set_kv(text, nkey, new_n)
        text = set_kv(text, Lkey, new_L)
        text = set_kv(text, xkey, flen[d])

    text = set_kv(text, "SimulationName", sim_name)
    return text, flen


def make_slurm(base_text: str, nodes: int, gpus_per_node: int, cpus_per_task: int,
                account: str, time_limit: str, run_dir: Path, inputfile: Path,
                logfile: Path, executable: Path) -> str:
    text = base_text
    text = re.sub(r"^(#SBATCH --nodes=)\S+", rf"\g<1>{nodes}", text, flags=re.MULTILINE)
    text = re.sub(r"^(#SBATCH --ntasks-per-node=)\S+", rf"\g<1>{gpus_per_node}", text, flags=re.MULTILINE)
    text = re.sub(r"^(#SBATCH --gpus-per-node=)\S+", rf"\g<1>{gpus_per_node}", text, flags=re.MULTILINE)
    text = re.sub(r"^(#SBATCH --cpus-per-task=)\S+", rf"\g<1>{cpus_per_task}", text, flags=re.MULTILINE)
    text = re.sub(r"^(#SBATCH --account=)\S+", rf"\g<1>{account}", text, flags=re.MULTILINE)
    text = re.sub(r"^(#SBATCH -t )\S+", rf"\g<1>{time_limit}", text, flags=re.MULTILINE)
    text = re.sub(r"^(#SBATCH -o )\S+", rf"\g<1>{run_dir / 'stdout'}", text, flags=re.MULTILINE)
    text = re.sub(r"^(#SBATCH -e )\S+", rf"\g<1>{run_dir / 'stderr'}", text, flags=re.MULTILINE)
    text = re.sub(r"^executable=\S+", f"executable={executable}", text, flags=re.MULTILINE)
    text = re.sub(r"^inputfile=\S+", f"inputfile={inputfile}", text, flags=re.MULTILINE)
    text = re.sub(r"^output_file=\S+", f"output_file={logfile}", text, flags=re.MULTILINE)
    # sbatch only parses #SBATCH directives up to the first non-comment line,
    # so the cd must go after the last one, not right after the shebang.
    lines = text.splitlines(keepends=True)
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith("#SBATCH"):
            insert_at = i + 1
    lines.insert(insert_at, f"\ncd {run_dir}\n")
    return "".join(lines)


def generate(args):
    base_input_text = DEFAULT_BASE_INPUT.read_text()
    base_slurm_text = DEFAULT_BASE_SLURM.read_text()

    submit_all = []
    for nodes in range(1, args.max_nodes + 1):
        run_dir = OUT_ROOT / f"N{nodes}"
        (run_dir / "data").mkdir(parents=True, exist_ok=True)

        sim_name = f"weakGEM3D_N{nodes}"
        inp_text, flen = make_input(base_input_text, nodes, args.gpus_per_node,
                                     args.per_gpu_cells, sim_name)
        inputfile = run_dir / f"{sim_name}.inp"
        inputfile.write_text(inp_text)

        logfile = run_dir / f"{sim_name}.log"
        slurm_text = make_slurm(base_slurm_text, nodes, args.gpus_per_node,
                                 args.cpus_per_task, args.account, args.time,
                                 run_dir, inputfile, logfile, args.executable)
        slurmfile = run_dir / f"submit_N{nodes}.slurm"
        slurmfile.write_text(slurm_text)
        slurmfile.chmod(0o755)

        total_ranks = nodes * args.gpus_per_node
        print(f"N={nodes:>2} nodes  ranks={total_ranks:>4}  "
              f"topology={flen['x']}x{flen['y']}x{flen['z']}  -> {slurmfile.relative_to(REPO_ROOT)}")
        submit_all.append(slurmfile)

    submit_script = OUT_ROOT / "submit_all.sh"
    with submit_script.open("w") as f:
        f.write("#!/bin/bash\nset -e\n")
        for s in submit_all:
            f.write(f"sbatch {s}\n")
    submit_script.chmod(0o755)
    print(f"\nGenerated {len(submit_all)} run directories under {OUT_ROOT.relative_to(REPO_ROOT)}/")
    print(f"Review the .inp/.slurm files, then submit with: {submit_script.relative_to(REPO_ROOT)}")

    if args.submit:
        for s in submit_all:
            subprocess.run(["sbatch", str(s)], check=True)


CYCLE_RE = re.compile(r"Execution time cycle:\s*([\d.]+)\s*ms")


def collect(args):
    rows = []
    for run_dir in sorted(OUT_ROOT.glob("N*"), key=lambda p: int(p.name[1:])):
        nodes = int(run_dir.name[1:])
        logs = list(run_dir.glob("*.log"))
        if not logs:
            continue
        times = [float(m) for m in CYCLE_RE.findall(logs[0].read_text())]
        if len(times) <= 1:
            continue
        times = times[1:]  # drop cycle 0 (warm-up / first-touch)
        mean_ms = sum(times) / len(times)
        ranks = nodes * args.gpus_per_node
        rows.append((nodes, ranks, mean_ms))

    if not rows:
        print("No completed logs found under", OUT_ROOT)
        return

    base_ms = rows[0][2]
    csv_path = OUT_ROOT / "results.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["nodes", "ranks", "mean_ms_per_cycle", "parallel_efficiency_pct"])
        print(f"{'nodes':>5} {'ranks':>6} {'ms/cycle':>10} {'efficiency':>11}")
        for nodes, ranks, mean_ms in rows:
            eff = 100.0 * base_ms / mean_ms
            w.writerow([nodes, ranks, f"{mean_ms:.3f}", f"{eff:.1f}"])
            print(f"{nodes:>5} {ranks:>6} {mean_ms:>10.2f} {eff:>10.1f}%")
    print(f"\nWrote {csv_path.relative_to(REPO_ROOT)}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-nodes", type=int, default=4, help="highest node count to generate (default: 4)")
    p.add_argument("--gpus-per-node", type=int, default=8)
    p.add_argument("--cpus-per-task", type=int, default=6)
    p.add_argument("--per-gpu-cells", type=int, default=64,
                    help="cells per GPU per dimension; kept constant for weak scaling (default: 64)")
    p.add_argument("--account", default="project_465002552")
    p.add_argument("--time", default="00:18:00", help="slurm -t time limit per job")
    p.add_argument("--executable", type=Path, default=DEFAULT_EXECUTABLE)
    p.add_argument("--submit", action="store_true", help="sbatch each job immediately after generating it")
    p.add_argument("--collect", action="store_true", help="summarize results from existing run logs instead of generating")
    args = p.parse_args()

    if args.collect:
        collect(args)
    else:
        generate(args)


if __name__ == "__main__":
    main()
