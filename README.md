## Gram Schmidt
Read about the process here. The idea is to use the recursive Gram-Schmidt process to orthonormalize a set of vectors.

https://en.wikipedia.org/wiki/Gram%E2%80%93Schmidt_process

### Solution - Recursive Method
Implemented a classical and modified recursive method for this process, the latter of which is designed to eliminate roundoff errors.

### Run Me
Run with a basic python 2/3 interpreter. Requires only numpy. Output shows some tests conjured up to verify that it is all working. Import into current workspace to orthonormalize your own vector set if you're so inclined.

```
>>> python gramSchmidt.py
Test 1: Finding orthonormal basis for simple vector space [[3, 1], [2, 2]]

For vector space provided [[3, 1], [2, 2]]
    Orthonormalized basis produced with classical Gram Schmidt process
[[ 0.9486833   0.31622777]
 [-0.31622777  0.9486833 ]]

    Orthonormal basis produced with Modified Gram Schmidt process:
[[ 0.9486833   0.31622777]
 [-0.31622777  0.9486833 ]],
Are the two about equal?? True
    Are all basis vectors orthagonal to eachother? True


Test 2: Finding orthonormal basis for arbritrary and more complex vector space [[3, 13, 2, 5], [1, 1, 2, 2], [8, -1, -0.5, 0], [1, -9, 0, 0]]
    For vector space provided
[[3, 13, 2, 5], [1, 1, 2, 2], [8, -1, -0.5, 0], [1, -9, 0, 0]]
Orthonormalized basis produced with modified Gram Schmidt process
[[ 0.20851441  0.90356246  0.13900961  0.34752402]
 [ 0.23774301 -0.37185445  0.71932501  0.53644577]
 [ 0.94667407 -0.11721351 -0.25246829 -0.16226199]
 [-0.0617106  -0.17765173 -0.63206616  0.75174732]]
    Are all basis vectors orthagonal to eachother? True
```
