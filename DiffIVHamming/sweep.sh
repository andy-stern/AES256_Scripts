#!/bin/bash

for ((NR=1; NR<=100; NR++)); do
    echo "NR=$NR"
    ./aes "$NR" 
done