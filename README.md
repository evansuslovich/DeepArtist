# MoMA-ML

Paintings.py: MetObjects.txt --> MetObjects.json --> MetObjects.csv

RGB.py: Goes through each image in folder and gets most common rgb value and adds to a csv (whatever you'd like to name it ... I named it output.csv)
**UPDATE**
RGB2.py: Goes through each folder in archive and fetches all imnages, iterates over the first 800 and gets rgb using more efficient method(20-30 percent (tested))

3d-rgb.py: Visualizing rgbs based off csv file from RGB.py 
