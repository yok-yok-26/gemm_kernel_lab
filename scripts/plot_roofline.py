#!/usr/bin/env python3
import csv, math, os, sys
from pathlib import Path
from simple_png import Canvas, text

csv_paths = [Path(x) for x in sys.argv[1:]] or [Path('reports/benchmark/library_cublas_gemm_latest.csv')]
out_dir = Path('reports/roofline')
out_dir.mkdir(parents=True, exist_ok=True)
rows = []
for csv_path in csv_paths:
    if csv_path.exists():
        with csv_path.open() as f:
            rows.extend(csv.DictReader(f))
if not rows:
    print('no benchmark rows')
    sys.exit(1)
points = []
for r in rows:
    m, n, k = int(r['m']), int(r['n']), int(r['k'])
    flops = 2 * m * n * k
    logical_bytes = 4 * (m * k + k * n + m * n)
    points.append((r['mode'], flops / logical_bytes, float(r['tflops']), float(r['ms'])))
peak_tflops = float(os.environ.get('PEAK_TFLOPS_EST', '30'))
peak_gbs = float(os.environ.get('PEAK_GBS_EST', '672'))
note = [f'{idx+1}. mode={mode}, logical_ai={ai:.6g}, tflops={tflops:.6g}, ms={ms:.6g}'
        for idx, (mode, ai, tflops, ms) in enumerate(points)]
for name, global_view in [('latest', False), ('global_latest', True)]:
    can = Canvas(); left, right, top, bottom = 80, 900, 55, 525
    can.rect(left, top, right, bottom, (0, 0, 0)); text(can, 90, 18, 'GEMM LOGICAL ROOFLINE', (0, 0, 0), 3)
    xs = [p[1] for p in points] + [peak_tflops * 1000 / peak_gbs]
    xmin = 0.5 if global_view else max(0.1, min(xs) * 0.7)
    xmax = 128 if global_view else max(xs) * 1.5
    ymin = 0.01
    ymax = max(peak_tflops * 1.2, max(p[2] for p in points) * 1.5, 1.0)
    lx0, lx1 = math.log10(xmin), math.log10(xmax); ly0, ly1 = math.log10(ymin), math.log10(ymax)
    def px(x): return left + (math.log10(max(x, xmin)) - lx0) / (lx1 - lx0) * (right - left)
    def py(y): return bottom - (math.log10(max(y, ymin)) - ly0) / (ly1 - ly0) * (bottom - top)
    prev = None
    for i in range(120):
        x = 10 ** (lx0 + (lx1 - lx0) * i / 119)
        y = min(peak_tflops, x * peak_gbs / 1000.0)
        cur = (px(x), py(y))
        if prev: can.line(prev[0], prev[1], cur[0], cur[1], (40, 90, 180))
        prev = cur
    colors = [(200, 40, 40), (20, 140, 80), (180, 120, 0), (120, 50, 180)]
    for idx, (mode, ai, tflops, ms) in enumerate(points):
        c = colors[idx % len(colors)]
        can.circle(px(ai), py(max(tflops, 1e-6)), 6, c)
        text(can, int(px(ai)) + 8, int(py(max(tflops, 1e-6))) - 8, str(idx + 1), c, 2)
    text(can, 82, 535, 'X FLOP/LOGICAL BYTE', (0, 0, 0), 2); text(can, 20, 55, 'Y TFLOP/S', (0, 0, 0), 2)
    out = out_dir / f'gemm_roofline_{name}.png'
    can.save(out)
    print(out)
(out_dir / 'gemm_roofline_latest.md').write_text(
    '# GEMM roofline note\n\n'
    'Initial scaffold plot uses benchmark FLOPs and logical bytes. Replace with NCU measured DRAM bytes and measured kernel duration after NCU exports are available for strict hardware roofline analysis. Peak ceilings are estimates unless overridden by PEAK_TFLOPS_EST and PEAK_GBS_EST.\n\n'
    + '\n'.join(note) + '\n')
print(out_dir / 'gemm_roofline_latest.md')
