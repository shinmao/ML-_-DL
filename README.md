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
