# cat namuwiki.00.txt | awk 'BEGIN {srand()} !/^$/ {if (rand() <= 0.1) print $0}' > sample.txt
# 실행 커맨트 : python3 trainer.py --tokenizer "wordpiece" --files "./corpus/namuwiki.00.txt"

from pathlib import Path
import subprocess
import time
import re

def shuf_sampling(source_path, output_path, rate, get_morph):
    if isinstance(source_path, Path):
        file_path = str(source_path).strip()
    
    p = subprocess.run([f"""wc -l {source_path}"""], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    input_size= float(re.findall('\d+', p.stdout)[0])
    sample_size = int(input_size * float(rate))
    # print(f"get {sample_size} sample from {int(input_size)} inputs")
    if get_morph:
        subprocess.run([f"""shuf -n {sample_size} {source_path} | mecab -O wakati -b 81920 > {output_path}"""], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    else:
        subprocess.run([f"""shuf -n {sample_size} {source_path} > {output_path}"""], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)