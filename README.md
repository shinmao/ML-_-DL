<<<<<<< HEAD
# ML-_-DL
<<<<<<< HEAD
 TEST ONE
 
=======

## ML
* What is machine learning?  
通過觀察大量的data，並且發現規律，來解決問題

* Learning: supervised learning v.s. unsupervised learning?  
有label v.s. 沒有label，我們的training set有沒有answer

* problem categories of supervised learning?  
classification problem, regression problem

* problem categories of unsupervised learning?  
clustering, non-clustering

* machine learning v.s. data mining?  
ML: use data to compute hypothesis g which approximates f.  
DM: use huge data to find interesting properties.  

* machine learning v.s. A.I.?  
ML只是A.I.其中一個領域

* machine learning v.s. statistics?  
統計是ML的重要工具 :cry:

### How to answer yes or no?
模型選擇：從hypothesis set中找到對的hypothesis
* 透過PLA演算法(Perceptron Learning Algorithm)來分類  
![](./screenshot/PLA1.png)  
上圖中是我們的classifier  
`wt + 1`為修正後的權重，`t`則是代表第幾輪  

> 深入探討PLA  

PLA是通過不斷內積來修正錯誤，運用的觀念是任何原點到data point的向量與我們hypothesis線的法向量做內積後：不是`+`就是`-`。

> PLA怎麼停下來？  

有點複雜又有趣的證明，有空再貼一下  
每一次修正的`wt+1`都會離目標權重`wf`更近，數學觀念上就是內積越來越大  
我們可能會懷疑內積越來越大不是因為兩線的夾角變小，而是因為向量的長度變大  
但通過證明發現，我們的mistake反而限制了我們向量長度的生長，因此長度對內積的影響並不會大到哪裡去  
而且根據公式，`兩個向量的夾角大於等於次數的根號與一個常數的product`  
因此T是有上限的!

> PLA的優點 v.s. 缺點  

優點: 容易實現，各種維度都行  
缺點: 需要linear separable

> 把non linear separable的情況是為noise

計算能夠最小化mistake的公式嗎？  
可惜...這是個NP-hard問題

> 用pocket演算法解決not linear separable的問題  

每一次都把目前mistake最少的線放在口袋裡  
碰到更好的就更新  
不要求完美，只要求最好

> 那我們怎麼不直接跑pocket呢？  

當然因為比較慢啊

* update...

## Refer
* 林軒田 - 機器學習基石
* 林軒田 - 機器學習技法
>>>>>>> e493aba4038a98548c4562021a393691cf0865dd
=======
### TYPES OF LEARNING
監督式學習

	都有輸出
	
半監督式學習
		
	一點輸出

非監督式學習
	
	無特定輸出
	
REINFORCEMENT LEARNING 增強式學習
	
	OUTPUT 逞罰/獎勵 (序列發生)
	E.G. 廣告系統(顧客是否點及)、玩牌系統

BATCH LEARNING 批量學習 (DUCK FEEDING)

	SPAM, PATIENT, CACSER, COIN 
	A VERRY COMMOM PROTOOL

ONLINE LEARNING 線上學習 (PASSIVE SEQUENTIAL)

	PLA, REINFORCEMENT LEARNING
	IMPROVES HYPOTHESIS

ACTIVE LEARNING (QUESTION ASKING) 

	SEQUENTIAL

##### CONCRETE FEATURE 

RAW FEATURES

RATING PREDICTION PROBLEM

NO PHYSICAL MWANING 


### Feasibility of learning 
大數法則
In sample error is close to out of sample error


Ein(g)≈Eout(g)
Ein(g)足夠小
Make sure that Eout(g) is close enough to Ein(g)


Effective Number of Line
Union bound ≈ unlimited (over-estimating)
線性可分 (shatter)
Must <=2^n 
n>m <<2^n 
 

	positive rays:   N+1 2

	positive intervals:  1/2 N^2 + 1/2N^1+ 1/2N^0<<2^N 3

	convex set : 2^N no

	2D:  <2^n 4

Break point

### Theory of Generalization
B(N,K)

### The VC Dimension  

Tatget 

	decrease model complexity dvc
	
	increase data size
	
	increase G error tolerance (寬容度)
	


Theory N ≈ 100,000*dvc

Prctice N ≈ 10d*vc
>>>>>>> 6bb5fa1addaf16eae872bceaedd0e3fedbba9450
