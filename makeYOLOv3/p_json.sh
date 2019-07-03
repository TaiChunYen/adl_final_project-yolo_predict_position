#!/bin/bash

files=$(ls $1)
for filename in $files
do
    python3.6 printjson.py -i $1/$filename
done
