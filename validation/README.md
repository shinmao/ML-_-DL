## validation
在maximum likelihood的方法中，很容易出現overfit的問題，因此我們要好好控制model的複雜度。  
如果dataset夠大，那model selection很簡單，拿一部分的data就可以train出一系列的model，通過獨立的test set選擇model即可。  
如果有限的data，迭代多次會導致overfit，所以以保留第三方的test set就變得很重要。那個第三方test set將會用作最後評估效能的工具。  
基於現實環境中dataset總是有限的難題，解決方法便是cross validation。
* k-fold cross validation: 總共k次，每一次我們都用 (k-1)/k 的資料來train
如果資料數量實在有限，那我們直接把k當資料的數量，這又叫leave one out。  
我們常用4-fold cross validation，每一次都用3/4的資料來train。  
缺點便是我們執行的次數隨著k上升  
* ...


# Which model to use (selection problem)
## How to choose model from many models?  
* The best Eout

### Model selection (Goal)
* Give: M models with H1, H2, H3… Hm with A1, A2, A3… Am. 
* Get: Select Hm that Gm=Am is the lowest Eout (The best model!!!)
* Fact: We do not know the best Eout!!

### How? Visually? CAUSE NOT!!
If we do not know the best Eout how about choose the best Ein
<img src="../screenshot/validation 1.png">

Ein <- Probability that H(n) and yn are not equal

### What can we do? 
 
1. Choose the best feature

	Feature 1126 always better than Feature 1 .right~

	λ= 0 is always better than λ= 0.1 
	
		=> OVERFITTING!!

2.	Compare models

	A1 minimizes Ein over H1

	A2 minimizes Ein over H2

	VC Dimension = c(H1∪H2) <- INCREASE MODEL COMPACITY

		Use Ein is dangerous!!!  Ein IS BAD!!
 
### How about select by the best Etest?
<img src="../screenshot/validation 2.png" width="40%">

What can you get from this method?

According finite-bin Hoeffding. 

<a href="https://www.codecogs.com/eqnedit.php?latex=Eout(g_{m}^{*})\leq&space;Etest(g_{m}^{*})&plus;O(\sqrt{\frac{logM}{Ntest}})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Eout(g_{m}^{*})\leq&space;Etest(g_{m}^{*})&plus;O(\sqrt{\frac{logM}{Ntest}})" title="Eout(g_{m}^{*})\leq Etest(g_{m}^{*})+O(\sqrt{\frac{logM}{Ntest}})" /></a>

Yes! We can get the best one But where is the Etest?
 
### Compare the method we already talk
1. Ein
* Calculated from dataset
* Feasible on hand
* Us the dataset that already used by Am
2. Etest 
* Calculated from test data
* Infeasible
* Clean!! But nowhere to get the Etest

3. A NEW WAY => E validation
* Calculated from Dval
* Feasible on hand
* Clean!!

### Validation Set Dval

<img src="../screenshot/validation 3.png"  width="70%">

Select K examples from D at random
Generalization guarantee for all M:

<a href="https://www.codecogs.com/eqnedit.php?latex=Eout(g_{m}^{-})\leq&space;Etest(g_{m}^{-})&plus;O(\sqrt{\frac{logM}{Ntest}})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Eout(g_{m}^{-})\leq&space;Etest(g_{m}^{-})&plus;O(\sqrt{\frac{logM}{Ntest}})" title="Eout(g_{m}^{-})\leq Etest(g_{m}^{-})+O(\sqrt{\frac{logM}{Ntest}})" /></a>

<img src="../screenshot/validation 4.png"  width="60%">
	 
After getting the best <a href="https://www.codecogs.com/eqnedit.php?latex=g_{m}^{-}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g_{m}^{-}" title="g_{m}^{-}" /></a> we can use the whole dataset to get the best <a href="https://www.codecogs.com/eqnedit.php?latex=g_{m}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?g_{m}" title="g_{m}" /></a>.
 
### Does that work?
 <img src="../screenshot/validation 5.png" width="40%">
 
We can observe that blue line performed better that red and in-sample line.
But why the red line is worse than in_sample line?
	Because the Gm- use the less data eventually.

### The dilemma of K
  <img src="../screenshot/validation 6.png"  width="80%">

Large K:

Make Eval closer to Eout(g-), but the Eout(g-) is far from Eout(gm)

Small K
	Make Eout(g) closer to Eout(g-), but the Eout(g-) is far from Eval.
	

## In practice:	<a href="https://www.codecogs.com/eqnedit.php?latex=K=\frac{N}{5}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?K=\frac{N}{5}" title="K=\frac{N}{5}" /></a>

### Extreme Cass: What if K =1?

Take K=1, <a href="https://www.codecogs.com/eqnedit.php?latex=D_{val}^{(n)}&space;=&space;{(Xn,Yn)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?D_{val}^{(n)}&space;=&space;{(Xn,Yn)}" title="D_{val}^{(n)} = {(Xn,Yn)}" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=E_{val}^{(n)}g_{n}^{-}&space;=&space;err(g_{n}^{-}(Xn),Yn)=&space;e_{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E_{val}^{(n)}g_{n}^{-}&space;=&space;err(g_{n}^{-}(Xn),Yn)=&space;e_{n}" title="E_{val}^{(n)}g_{n}^{-} = err(g_{n}^{-}(Xn),Yn)= e_{n}" /></a>

Average over possible<a href="https://www.codecogs.com/eqnedit.php?latex=E_{val}^{(n)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E_{val}^{(n)}" title="E_{val}^{(n)}" /></a><= leave-one-out cross validation

<a href="https://www.codecogs.com/eqnedit.php?latex=E_{loocv}(H,A)&space;=&space;\frac{1}{N}\sum_{n=1}^{N}e_{n}&space;=&space;\frac{1}{N}\sum_{n=1}^{N}err(g_{n}^{-}(X_{n}),Y_{n})" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E_{loocv}(H,A)&space;=&space;\frac{1}{N}\sum_{n=1}^{N}e_{n}&space;=&space;\frac{1}{N}\sum_{n=1}^{N}err(g_{n}^{-}(X_{n}),Y_{n})" title="E_{loocv}(H,A) = \frac{1}{N}\sum_{n=1}^{N}e_{n} = \frac{1}{N}\sum_{n=1}^{N}err(g_{n}^{-}(X_{n}),Y_{n})" /></a> 

Is that means anything?

  <img src="../screenshot/validation 7.png"  width="60%">

## Our goal is to make  	<a href="https://www.codecogs.com/eqnedit.php?latex=E_{loocv}(H,A)&space;\approx&space;E_{out}(g)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E_{loocv}(H,A)&space;\approx&space;E_{out}(g)" title="E_{loocv}(H,A) \approx E_{out}(g)" /></a> and here we are

### Leave-One-Out Cross Validation in Practice

  <img src="../screenshot/validation 8.png"  width="60%">
   <img src="../screenshot/validation 9.png"  width="50%">

We can observe that Eloocv performs better than Ein and close to Eout.

### But… 
If our N is 10,000?? We must do 10,000 times computation.

V-fold cross-validation: Random-Partition D to V equal parts and taking V-1 for training and 1 for validation.

<img src="../screenshot/validation 10.png"  width="60%">
	


 
In practice, V =10

But be careful…
<img src="../screenshot/validation 11.png"  width="60%">