# cat namuwiki.00.txt | awk 'BEGIN {srand()} !/^$/ {if (rand() <= 0.1) print $0}' > sample.txt
# 실행 커맨트 : python3 trainer.py --tokenizer "wordpiece" --files "./corpus/namuwiki.00.txt"

from pathlib import Path
import subprocess
import time


def execute_script(file_path, rate, target="corpus", replace="sample"):
    """get sample from text file using shell"""
    if isinstance(file_path, str):
        file_path = str(file_path)

    output_path = str(file_path).replace(target, replace)
    sampler = f"""cat {file_path} | awk 'BEGIN {{srand()}} !/^$/ {{if (rand() <= {rate}) print $0}}' > {output_path}"""
    subprocess.run([sampler], shell=True, check=True)
