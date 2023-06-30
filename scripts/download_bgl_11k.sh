#!/bin/bash

file="${HOME}/.dataset/bgl/"
if [ -e $file ]
then
  echo "$file exists"
else
  mkdir -p $file
fi

wget https://raw.githubusercontent.com/salesforce/logai/main/examples/datasets/BGL_AD/BGL_11k.log -P $file
