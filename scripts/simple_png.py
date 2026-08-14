#!/usr/bin/env python3
import struct, zlib

class Canvas:
    def __init__(self, w=960, h=600, bg=(255,255,255)):
        self.w=w; self.h=h; self.p=[list(bg)*w for _ in range(h)]
    def set(self,x,y,c):
        if 0<=x<self.w and 0<=y<self.h: self.p[y][3*x:3*x+3]=list(c)
    def line(self,x0,y0,x1,y1,c=(0,0,0)):
        x0=int(round(x0)); y0=int(round(y0)); x1=int(round(x1)); y1=int(round(y1))
        dx=abs(x1-x0); sx=1 if x0<x1 else -1; dy=-abs(y1-y0); sy=1 if y0<y1 else -1; err=dx+dy
        while True:
            self.set(x0,y0,c)
            if x0==x1 and y0==y1: break
            e2=2*err
            if e2>=dy: err+=dy; x0+=sx
            if e2<=dx: err+=dx; y0+=sy
    def rect(self,x0,y0,x1,y1,c,fill=False):
        if fill:
            for y in range(max(0,int(y0)), min(self.h,int(y1)+1)):
                for x in range(max(0,int(x0)), min(self.w,int(x1)+1)): self.set(x,y,c)
        else:
            self.line(x0,y0,x1,y0,c); self.line(x1,y0,x1,y1,c); self.line(x1,y1,x0,y1,c); self.line(x0,y1,x0,y0,c)
    def circle(self,cx,cy,r,c):
        for y in range(int(cy-r), int(cy+r)+1):
            for x in range(int(cx-r), int(cx+r)+1):
                if (x-cx)*(x-cx)+(y-cy)*(y-cy)<=r*r: self.set(x,y,c)
    def save(self,path):
        raw=b''.join(b'\x00'+bytes(row) for row in self.p)
        def chunk(t,d): return struct.pack('>I',len(d))+t+d+struct.pack('>I',zlib.crc32(t+d)&0xffffffff)
        data=b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',self.w,self.h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw,9))+chunk(b'IEND',b'')
        open(path,'wb').write(data)

FONT={
 '0':['111','101','101','101','111'], '1':['010','110','010','010','111'], '2':['111','001','111','100','111'], '3':['111','001','111','001','111'],
 '4':['101','101','111','001','001'], '5':['111','100','111','001','111'], '6':['111','100','111','101','111'], '7':['111','001','010','010','010'],
 '8':['111','101','111','101','111'], '9':['111','101','111','001','111'], '.':['0','0','0','0','1'], '-':['0','0','1','0','0'],
 'A':['010','101','111','101','101'], 'B':['110','101','110','101','110'], 'C':['111','100','100','100','111'], 'D':['110','101','101','101','110'],
 'E':['111','100','110','100','111'], 'F':['111','100','110','100','100'], 'G':['111','100','101','101','111'], 'H':['101','101','111','101','101'],
 'I':['111','010','010','010','111'], 'K':['101','101','110','101','101'], 'L':['100','100','100','100','111'], 'M':['101','111','111','101','101'],
 'N':['101','111','111','111','101'], 'O':['111','101','101','101','111'], 'P':['111','101','111','100','100'], 'R':['110','101','110','101','101'],
 'S':['111','100','111','001','111'], 'T':['111','010','010','010','010'], 'U':['101','101','101','101','111'], 'V':['101','101','101','101','010'],
 'X':['101','101','010','101','101'], 'Y':['101','101','010','010','010'], '/':['001','001','010','100','100'], ':':['0','1','0','1','0'],
 ' ':['0','0','0','0','0'], '_':['0','0','0','0','111']
}
def text(can,x,y,s,c=(0,0,0),scale=2):
    cx=x
    for ch in str(s).upper():
        pat=FONT.get(ch, FONT[' '])
        for yy,row in enumerate(pat):
            for xx,v in enumerate(row):
                if v=='1': can.rect(cx+xx*scale,y+yy*scale,cx+(xx+1)*scale-1,y+(yy+1)*scale-1,c,True)
        cx += (len(pat[0])+1)*scale
