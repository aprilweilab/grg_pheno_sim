#!/bin/bash
set -ev

flake8 grg_pheno_sim/ --count --select=E9,F63,F7,F82,F401 --show-source --statistics

black grg_pheno_sim/ setup.py test/ --check 

pytest -x test/

