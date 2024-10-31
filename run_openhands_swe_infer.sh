#!/bin/bash

# conda create -n openhands_env python=3.12 -c conda-forge

sh ./evaluation/swe_bench/scripts/run_infer.sh llm.llama-3.1-8b-instruct HEAD CodeActSWEAgent 1 30 1 princeton-nlp/SWE-bench_Lite test