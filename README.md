# ML-_-DL

## Applied Machine Learning Course

#### Tool: MATLAB

* 總結課程教學: 
  * 第一堂:了解Machine Learning基本知識，並講解不同學習法(Supervised, Unsupervised, Reinforcement)的不同功用，使用MATLAB作為實作工具。
  * 第二堂:線代。監督學習:KNN
  * 第三堂:Decision Tree，介紹他的演算法、bias、衍伸問題
  * 第四堂:Bias和方差的權衡，線性模型應用
  * 第五堂:機器學習的應用
  * 第六堂:ANN，進階ANN:多層感知器和反向傳播(Multi-Layers Perceptron and Backpropagation)
  * 第七堂:Logistic Regression
  * 第八堂:Ensemble Classifiers:Bootstrap, Bagging, Boosting and Adaboost
  * 第九堂:網路安全(Generative Adversarial Networks and Applications in Cyber Security.)
  * 第十堂:Bayesian Learning
  * 第十一堂:Unsupervised learning: clustering, K-means, EM(會有GMM在多生物系統中用於密度估計的應用)
  * 第十二堂:PCA臉部辨識
  * 第十三堂:Reinforcement Learning, Markov Models(強化學習，馬爾可夫模型)
  * 第十四堂:Deep learning:CNN(CNN在生物識別系統中的反欺騙應用),Transfer learning




#### Leture 1

**_Automatic discovery of regularities in data(我們要讓機器自動發現數據中的規律性)_**


The major focus is to extract information from data automatically, by computational and statstical methods. Machine learning is the design of algorithms that can improve their performance (given some quantifiable measure) with experience. Machine can use past experience to improve future performance and modify its learning from the knowledge we give (big data). 


Applications: Natural Language Processing, Search Engines, Medical Diagnosis, Bioinformatics, Stock Market Analysis

Generalization is key: it is easy to learn by heart, difficult to learn general-purpose strategies
innate VS acquired knowledge (provide data to the machine, and machine will learn by itself)

###### 科普:何謂「泛化（Generalization）」，意思是我們的設計模組可以應對未來的數據，也就是可以被廣泛使用卻仍在我們模型範圍內，也就是他的適應性很好，這就是泛化。(模型很好地擬合以前未見過的新數據)

###### 我們在訓練模型的時候，會調整一些超參數來讓我們的模型可以適應其他狀況。但現實是，有時候我們誤差值很小，但仍然無法適應新的數據，這就是我們模型過度擬合了其訓練數據的特性。 

###### 如何知道該模型已泛化：
###### 1.模型的複雜性      2.模型在訓練數據上的表現

###### Ref:[泛化](https://ithelp.ithome.com.tw/articles/10221782?sc=iThelpR)
-------------------------------------------------------------------------------------------------
#### WHY use learning?

It is too difficult to design a set of rules by hand.


#### WHEN we use learning?
1. A pattern exists
2. We cannot pin it down mathematically
3. We have data on it
--------------------------------------------------------------------------------------------------
#### Task: What is our goal? What are we tring to do?
#### Experience: What data do we provide the algorithm?
#### Performance Metrics: How do we measure how well the system is doing?

---------------------------------------------------------------------------------------------------
#### Types of Learning
-Supervised Learning

-Unsupervise Learning

-Reinforcement Learning

#### Definition
What is "Representation", "Evaluation", and "Generalization"?


#### Leture 3 Types of Learning


| 模型名稱 | 適合數據類型 | 例子 | 
| ------------- | :-------------: | :----- |
| Binary Classification | 是非題 | 1,-1 |
| Multiclass Classification | 分種類 | 視覺辨識 |
| Regression | 連續數字 | 股票、溫度 |



| 學習名稱        | 解釋 | 例子  |
| ------------- |:-------------:|:-----|
| Supervised Learning | 每個x都有相對應的y| |
| Unsupervised Learning | 每個x都沒有相對應的y | 1.clustering 2.desity estimation 3.outlier detection |
| Semi-supervised Learning | 每個x不一定都有相對應的y | 1.FB的照片辨識朋友功能(有些人有label了) 2.藥效預測 (label比較昂貴時適用)|
| Reinforcement Learning | 對或錯的方式訓練機器 | 1.廣告系統(利用客戶反應訓練) 2.計算玩牌的勝算 |


| 協議方法名稱        | 解釋  |
| ------------- |:-----|
| Batch Learning (批量學習) | 填鴨式教育 |
| Online Learning | 被動學習 |
| Active Learning | 主動提問(用於label比較貴的數據) |


| 輸入名稱        | 解釋  |
| ------------- |:-----|
| Concrete Feature | 具體特徵，複雜且可能相關的描述 |
| Raw Feature | 沒有具體特徵，需要人或機器轉換成具體特徵，簡單的描述 |
| Abstract Feature | 沒有具體特徵，需要人或機器轉換成具體特徵，沒有描述 |


#### Leture 4 Feasibility of Learning

怎樣是好的學習?
* 好的學習:對於h而言Ein(h)是小的，且演算法選擇的h趨近於g ==> 'g=f' PAC
* 不好的學習:Ein通常不小，演算法被迫選擇h作為g ==> 'g!=f' PAC
* 實際情況:演算法自己做選擇H(ex:PLA),而非被迫選擇h

有很多的H，應該用哪種呢?
* 比較合理的演算法(PLA/pocket):選擇一個擁有最小Fin(Hm)當作g的Hm

只有一個H(hypothesis)的Bad Data:
*Eout(H)和Ein(H)差很多
有很多H的Bad Data:
*存在部分H會有Eout(H)和Ein(H)差很多的問題(演算法無法自由做選擇)

Learning到底可不可行?
假如|H|=M(hypothesis)是有限的，N(資料量)足夠大，不論哪個被演算法選中的g都會有，Ein(g)≈Eout(g)的結果
假如演算法找到一個Ein(g)≈0的g，PAC保證Eout(g)也會趨近於0 --->learning possible!

![圖]( https://github.com/shinmao/ML-_-DL/blob/dev2/1.jpg "圖片名稱")



#### Leture 5 Training V.S. Testing
在Ein(g)≈Eout(g)≈0 這樣的基礎下，Eout(g)≈0是機器在訓練的部分，而Ein(g)≈Eout(g)是測試這個model可不可行的階段

##### 為了衡量一下M(amount of hypothesis set)大比較好還是小，提出兩個問提:
- 1.我們可以確定Eout(g)靠近Ein(g)嗎?
- 2.Ein(g)他夠小嗎?

##### 在數量小的M:
- 1.Yes! 會發現M帶入P[BAD]=<2*M*exp(...)公式，因為M小，P[BAD]也會變小
- 2.No! 因為M小，選擇太少了


##### 在數量大的M:
- 1.No! P[BAD]增加
- 2.Yes! 因為M大，選擇很多，可以找到夠小的Ein(g)


![圖]( https://github.com/shinmao/ML-_-DL/blob/dev2/2.jpg "圖片名稱")
![圖]( https://github.com/shinmao/ML-_-DL/blob/dev2/3.jpg "圖片名稱")
![圖]( https://github.com/shinmao/ML-_-DL/blob/dev2/4.jpg "圖片名稱")




#### Leture 6 Theory of Generalization
![圖]( https://github.com/shinmao/ML-_-DL/blob/dev2/4.5.jpg "圖片名稱")


#### Leture 7 The VC Dimension 
![圖]( https://github.com/shinmao/ML-_-DL/blob/dev2/5.jpg "圖片名稱")
![圖]( https://github.com/shinmao/ML-_-DL/blob/dev2/6.jpg "圖片名稱")

##### M and dvc:

##### 為了衡量一下M(amount of hypothesis set)和dvc大比較好還是小，提出兩個問提:
- 1.我們可以確定Eout(g)靠近Ein(g)嗎?
- 2.Ein(g)他夠小嗎?

##### 在數量小的M:
- 1.Yes! 會發現M帶入P[BAD]=<2*M*exp(...)公式，因為M小，P[BAD]也會變小
- 2.No! 因為M小，選擇太少了


##### 在數量大的M:
- 1.No! P[BAD]增加
- 2.Yes! 因為M大，選擇很多，可以找到夠小的Ein(g)


##### 在數量小的dvc:
- 1.Yes! 會發現dvc帶入P[BAD]=<2*(2N)^dvc*exp(...)公式，因為dvc小，P[BAD]也會變小
- 2.No! 因為dvc小，限制太多


##### 在數量大的M:
- 1.No! P[BAD]增加
- 2.Yes! 很多權利

