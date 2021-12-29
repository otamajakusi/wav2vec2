import sys
import subprocess
import j2a
from pprint import pprint

wav = sys.argv[1]
txt = sys.argv[2]

txt_list = j2a.j2a(txt)
pprint(txt_list)
for i, t in enumerate(txt_list):
    start = t[0]
    stop = t[1]
    out = f"{i}.wav"
    cmd = f"ffmpeg -i {wav} -ss {start} -to {stop} -ac 1 -ar 16000 {out} -y -v quiet -hide_banner"
    subprocess.check_output(cmd, shell=True)

