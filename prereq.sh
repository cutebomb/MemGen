#!/bin/bash
set -eo pipefail

cd 
git clone https://github.com/cutebomb/MemGen.git

cd MemGen
pip install -r requirements.txt
pip install flash-attn --no-build-isolation