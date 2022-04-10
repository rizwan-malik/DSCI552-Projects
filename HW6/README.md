## Homework 6 - Support Vector Machines
Given two data files - **linsep.txt** and **nonlinsep.txt** - each of which contains 100 2D points with classification labels +1 or -1. The first two columns in each file indicate the 2D coordinates of a point; and the third column indicates its classification label. The points in **linsep.txt** are linearly separable. The points in **nonlinsep.txt** are not linearly separable in the original space but are linearly separable in a zspace that uses a simple nonlinear transformation.
### 1. Linearly Separable Data
   - Find the fattest margin line that separates the points in **linsep.txt**. Using a Quadratic Programming solver report the equation of the line as well as the support vectors.
### 2. Linearly Not Separable Data
- Using a kernel function along with the same Quadratic Programming solver, find the equation of a curve that separates the points in **nonlinsep.txt**. Report the kernel function as well as the support vectors.
