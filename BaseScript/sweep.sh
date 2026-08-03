#!/bin/bash

FIXEDSIZEMB=16
SIZEBYTES=$((FIXEDSIZEMB * 1024 * 1024 * 2))

run() {
    NR=$1
    echo "NR=$NR SIZE=${FIXEDSIZEMB}MB"
    ./aes "$NR" "$SIZEBYTES"
}

for ((NR=1; NR<=100; NR++)); do
    run "$NR"
done