#!/bin/bash

rm -rf chroma_db/*
cd ./tools || return
python create_database.py
rm output.txt
rm line_num.txt
cd .. || return
python main.py
cd ./tools || return
python evaluate.py -g ../../../dataset/BGL/BGL_2k.log -p ./output.txt -v
python create_rag_dataset.py
chmod 777 rag_dataset.txt