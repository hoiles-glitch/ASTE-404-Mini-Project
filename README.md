# ASTE-404-Mini-Project
A one-dimensional FTCS solver for the unsteady heat equation of the zenith facing panel of a LVLH spacecraft in Earth Orbit.

It involves a 1 meter thick metal wall with different coatings.
It produces:
- Wall temperatures after one day in orbit
- A trace of solar flux vs time in orbit

Structure
ASTE-404-Mini-Project/
   src/
      main.py
   results/
      
   report/
      .gitignore
      README.md

To run:
conda create -n miniproject python=3.10
conda activate miniproject
pip install -r numpy
pip install -r matplotlib
python -m src.main
