#!/usr/bin/env python3
import csv, sys
from pathlib import Path
from simple_png import Canvas, text
files=[Path(p) for p in sys.argv[1:]] or sorted(Path('reports/benchmark').glob('*_latest.csv'))
rows=[]
for f in files:
    if f.exists(): rows+=list(csv.DictReader(f.open()))
if not rows: print('no benchmark CSV rows'); sys.exit(1)
out=Path('reports/trends'); out.mkdir(parents=True, exist_ok=True)
can=Canvas(); left,right,top,bottom=90,900,60,520
can.rect(left,top,right,bottom,(0,0,0)); text(can,100,22,'GEMM RUNTIME TREND',(0,0,0),3)
ys=[float(r['ms']) for r in rows]; ymax=max(ys)*1.2 if ys else 1; ymin=0
colors=[(200,40,40),(20,140,80),(40,90,180),(160,90,20)]
for i,r in enumerate(rows):
    x=left+40+i*max(60,(right-left-80)//max(1,len(rows)-1)); y=bottom-(float(r['ms'])-ymin)/(ymax-ymin if ymax>ymin else 1)*(bottom-top)
    can.circle(x,y,5,colors[i%len(colors)]); text(can,x-18,bottom+12,str(i+1),(0,0,0),2); text(can,x+8,int(y)-8,r['mode'][:10],colors[i%len(colors)],2)
text(can,92,535,'CONFIG INDEX',(0,0,0),2); text(can,20,62,'MS',(0,0,0),2)
png=out/'gemm_benchmark_runtime.png'; can.save(png)
(out/'gemm_benchmark_runtime.md').write_text('Benchmark trend generated from archived CSV files.\n\n'+'\n'.join(f"{i+1}. {r['mode']} {r['m']}x{r['n']}x{r['k']} ms={float(r['ms']):.6g} tflops={float(r['tflops']):.6g}" for i,r in enumerate(rows))+'\n')
print(png)
