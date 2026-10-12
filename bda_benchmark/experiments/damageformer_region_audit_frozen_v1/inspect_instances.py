#!/usr/bin/env python3
import argparse,json
from .instance_utils import inspect_instance_dir
p=argparse.ArgumentParser(); p.add_argument('--instance_dir',required=True); p.add_argument('--limit',type=int,default=30); a=p.parse_args()
print(json.dumps(inspect_instance_dir(a.instance_dir,a.limit),indent=2,ensure_ascii=False))
