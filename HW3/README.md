
## Homework 3 - FastMap & PCA Implementation
1. Use **FastMap** to embed the objects in **fastmap-data.txt** into a 2D space.
   - The first two columns in each line of the data file represent the IDs of the two objects; and the third column indicates the symmetric distance between them.
   - The objects listed in **fastmap-data.txt** are actually the words in **fastmapwordlist.txt** (nth word in this list has an ID value of n).
   - The distance between each pair of objects is the **Damerau–Levenshtein** distance between them.
   #### Deliverable:
   Plot the words on a 2D plane using your FastMap solution.
   
2. Use **Principal Component Analysis** to reduce the dimensionality of the data points in PCA **pca-data.txt** from 3D to 2D.
   - Each line of the data file represents the 3D coordinates of a single point.
   #### Deliverable:
   Output the directions of the first two principal components.
