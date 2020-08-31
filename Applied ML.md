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
