import json
import os
import subprocess

dirn = os.path.dirname(os.path.abspath(__file__))
filename = dirn + '/daphne'


def daphne(args, cwd=filename):
    proc = subprocess.run(['lein','run','-f','json'] + args,
                          capture_output=True, cwd=cwd)
    if(proc.returncode != 0):
        raise Exception(proc.stdout.decode() + proc.stderr.decode())
    return json.loads(proc.stdout)

