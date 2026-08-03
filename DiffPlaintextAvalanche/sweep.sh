#!/bin/bash

for ((NR=51; NR<=100; NR++)); do
    echo "NR=$NR"
    ./aes "$NR" 
done