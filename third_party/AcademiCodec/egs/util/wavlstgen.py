# -*- encoding: utf-8 -*-
# 2022-2023 by zhaomingwork@qq.com
# can be used for generating train.lst or valid.lst only given a root dir
# example:
# python wavlstgen.py --wavdir /data/asr_data/aishell/ --outfile train.lst
import os
import time
 
import argparse
import json
import traceback
 

import logging

logging.basicConfig(level=logging.ERROR)

parser = argparse.ArgumentParser()
parser.add_argument("--wavdir",
                    type=str,
                    default="./",
                    required=True,
                    help="root dir of wav")
 

parser.add_argument("--outfile",
                    type=str,
                    default="./wav.lst",
                    required=False,
                    help="output list file name")

args = parser.parse_args()

print(args)

def genwavlist(rootdir):
  outlist = open(args.outfile, 'w+')
  
  for dirpath, dirnames, filenames in os.walk(rootdir):
     for filename in filenames:
        #print(os.path.join(dirpath, filename))
        if filename.endswith(".wav"):
            outlist.write(os.path.join(dirpath, filename)+"\n")
  outlist.close()


if __name__ == '__main__':
    
    genwavlist(args.wavdir)
