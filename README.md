# Sudoku Solver on real-time webcam feed!  

Taking a video input via a webcam and then extracting the Sudoku from it. Digits are extracted from this Sudoku which is then recognised using a model I trained on a very small Char74k Dataset(only digits).
___
![Image](https://github.com/imsahil007/SudokuSolver/blob/master/res/sol2.png)
![Gif](https://github.com/imsahil007/SudokuSolver/blob/master/res/output_1.gif)  

## Run:
```
python solveSudoku.py
```
Using a LeNet design for training my model. I tried using the MNIST dataset as well but the dissimilarity in '1' on printed characters doesn't allow it to work well.  

To solve the sudoku. I am using Peter Norvig's backtracking algorithm. And after finding the solution I am overlaying the solution(Basically the digits) on the video feed itself. You can also save the output both as image and video.  
The whole code is in Python. I followed various blogs for this. I will add the references in the comments section.  

**Note**: It might ask your webcam id. The code will print itself all available webcam IDs  

## Requirements:
* Python3
* open-cv
* Webcam

# References:
* [Norvig's algorithm for solving sudoku](https://norvig.com/sudoku.html)
* Sudoku Extraction: 
    1. [Part 1](https://medium.com/@neshpatel/solving-sudoku-part-i-7c4bb3097aa7) 
    2. [Part 2](https://medium.com/@neshpatel/solving-sudoku-part-ii-9a7019d196a2)  

* [LeNet Model](https://www.kaggle.com/cdeotte/25-million-images-0-99757-mnist)
* Dataset: [Chark74k](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/)
* [My MNIST Model with 99.34% accuracy](https://github.com/imsahil007/MNIST-LeNet)