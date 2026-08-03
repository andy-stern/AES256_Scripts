## Producing the results

### Requirements
- A C compiler (gcc used in this case) and Python 3.10+
- Linux or WSL2
- Python packages, `pip install -r requirements.txt`

### Compile the generators
- `BaseScript: gcc -O3 -march=native -funroll-loops aes.c -o aes`
- `DifferentialDelta: gcc -O3 -march=native -funroll-loops aes.c -o aes`
- `DiffIVHamming: gcc -O3 -march=native -funroll-loops aes.c -o aes`
- `DiffKeyHamming: gcc -O3 -march=native -funroll-loops aes.c -o aes`
- `DiffPlaintextAvalanche: gcc -O3 -march=native -funroll-loops -fopenmp aes.c -o aes`
- `DiffPlaintextHamming: gcc -O3 -march=native -funroll-loops aes.c -o aes`
- `IntegralSquare: gcc -O3 -march=native -funroll-loops aes.c -o aes`

### Generating the data
- For each folder, there is a script to generate data for NR 1-100: `sweep.sh`
- BaseScript's `sweep.sh` differs slightly, allowing to input the NR and amount of data it generates

### Generating all figures and reports
`python aes.py`
`python sbox.py`

Outputs appear in `reports/<experiment>/` with report.txt and figures. 
DDT / LAT heatmaps and sbox_report.txt in `sbox/`

### Notes
- Randomness is drawn from `/dev/urandom` at generation time
- BaseScript is included for completeness but contribute nothing to the results and analyses
