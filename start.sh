#!/bin/bash

cd detection || return
rm -rf chroma_db/*
rm output.txt
cd ../tools || return
python create_database.py
rm line_num.txt
cd ../detection || return
python main.py
cd ../tools || return
python evaluate.py -g ../dataset/BGL/BGL_2k.log -p ../detection/output.txt -v
python extract_db_dataset.py
chmod 777 rag_dataset.txt
