@echo off
call C:\Users\makhan.gill\Anaconda3\Scripts\activate.bat concussion
C:\Users\makhan.gill\Anaconda3\envs\concussion\python.exe -m jupyter nbconvert --execute --to notebook C:\Users\makhan.gill\Documents\GitHub\claims_trends\claims_trends\concussion_notebook.ipynb
call conda deactivate
C:\Users\makhan.gill\Anaconda3\Scripts\jupyter.exe nbconvert C:\Users\makhan.gill\Documents\GitHub\claims_trends\claims_trends\concussion_notebook.nbconvert.ipynb --to html --output C:\Users\makhan.gill\Documents\GitHub\claims_trends\output\output.html
