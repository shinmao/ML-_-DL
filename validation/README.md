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

Ein <- Probability that H(n) and yn are not equal

### What can we do? 
<img src="../screenshot/validation 1.png">

 
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
	 
  	What can you get from this method?
		According finite-bin Hoeffding. 
		Eout(gm*)≤Etest(gm*)+O(√(log⁡M/Ntest))
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

3. A NEW WAY~ <= E validation
* Calculated from Dval
* Feasible on hand
* Clean!!

Validation Set Dval
	Select K examples from D at random
Generalization guarantee for all M:
Eout(gm-)≤Etest(gm-)+O(√(log⁡M/Ntest)) 
	 
	After getting the best gm- we can use the whole dataset to get the best gm.
 
Does that work? 
 
We can observe that blue line performed better that red and in-sample line.
But why the red line is worse than in_sample line?
	Because the Gm- use the less data eventually.

The dilemma of K
 
Large K:
	Make Eval closer to Eout(g-), but the Eout(g-) is far from Eout(gm)
Small K
	Make Eout(g) closer to Eout(g-), but the Eout(g-) is far from Eval.
In practice:　K=N/5
Extreme Cass: What if K =1?
	Take K=1, D_val^((n))= {(Xn,Yn)} and E_val^((n))(gn-)=err(gn-(Xn),Yn) = En
	Average over possible E_val^((n))<= leave-one-out cross validation
	E_loocv (H,A)=  1/N ∑_(n=1)^N▒En . Is that means anything?
  
Leave-One-Out Cross Validation in Practice
 
   

But… If our N is 10,000?? We must do 10,000 times computation.
	V-fold cross-validation: 
Random-Partition D to V equal parts and taking V-1 for training and 1 for validation.
 
In practice, V =10
But be careful…
 
