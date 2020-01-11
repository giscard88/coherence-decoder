#! /bin/bash

python convert_tensor.py --duration 250 --channel 2
python convert_tensor.py --duration 250 --channel 3
python convert_tensor.py --duration 250 --channel 4
python convert_tensor.py --duration 250 --channel 5
python convert_tensor.py --duration 250 --channel 10

python convert_tensor.py --duration 500 --channel 2
python convert_tensor.py --duration 500 --channel 3
python convert_tensor.py --duration 500 --channel 4
python convert_tensor.py --duration 500 --channel 5
python convert_tensor.py --duration 500 --channel 10

python convert_tensor.py --duration 1000 --channel 2
python convert_tensor.py --duration 1000 --channel 3
python convert_tensor.py --duration 1000 --channel 4
python convert_tensor.py --duration 1000 --channel 5
python convert_tensor.py --duration 1000 --channel 10

python convert_tensor.py --duration 4000 --channel 2
python convert_tensor.py --duration 4000 --channel 3
python convert_tensor.py --duration 4000 --channel 4
python convert_tensor.py --duration 4000 --channel 5
python convert_tensor.py --duration 4000 --channel 10
