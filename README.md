This is a dissertation project

Configuration used 

->used github codespaces to run the project
->created a virtual env
    ->commands
    ->python3 -m venv ~/.venv  [python virtual env create it into home directory into an invisible directory]
    ->vim ~/.bashrc [source this into bash config file]
    ->shift +g 
    ->source venv
    ->source ~/.venv/bin/activate
    ->wq ->open a new window
    #Building scaffold
   touch Makefile
   touch requirements.txt
   touch main.py
   touch test_main.py

->copy the makefile template
->add requirements.txt
->pip freeze |less for code reusability and copy the versions

Continous integration step


These are action steps:

[![Test Multiple Python Versions](https://github.com/jayakvlr/silcon_veld/actions/workflows/main.yml/badge.svg)](https://github.com/jayakvlr/silcon_veld/actions/workflows/main.yml)
