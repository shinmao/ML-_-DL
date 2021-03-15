# ML-_-DL
- [ML-_-DL](#ml-_-dl)
  - [ML](#ml)
  - [ML數學概率基礎](#ml數學概率基礎)
  - [維度](#維度)
  - [學習方法](#學習方法)
  - [來談談我們不同種類的input](#來談談我們不同種類的input)
  - [Training versus Testing](#Training-versus-Testing)
  - [Theory of Generalization](#Theory-of-Generalization)
  - [The VC Dimension](#The-VC-Dimension)
  - [Noise and Error](#Noise-and-error)
  - [Validation](#Validation)
  - [學習的可行性](#學習的可行性)
  - [linear classification](#linear-classification)
  - [Linear Regression](#linear-regression)
    - [Generative model v.s. Discriminative model](#generative-model-vs-discriminative-model)
    - [SVM](#svm)
  - [DL](#dl)

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

## ML數學概率基礎
* [數學概率基礎](./Math-Probs-Statistics/README.md)

## 維度
* [維度災難 (Curse of dimensionality)](./dimension.md#維度災難)
* [PLA 降維](./dimension.md#pca-principal-component-analysis)

## 學習方法
| 方法        | 有沒有label | 描述  |
| ------------- |:-------------:|:-----|
| supervised learning | 全都有 | 二元分類，多元分類，回歸。按照輸出空間的話，還可以在區分出結構化學習(細節放在下面)。 |
| unsupervised learning | 沒有 | clustering, desity estimation, outlier detection |
| semi- | 有些有 | FB的照片辨識朋友功能(有些人有label了), 藥效預測 (label比較昂貴時適用) |
| reinforcement learning | 我們給的不是label | 寵物訓練的例子很好懂。我們給machine的output feedback，告訴他是對的還是不對的。最重要的是這個學習的過程是sequential的，machine會透過一筆一筆資料學習，而非我們一次餵大筆資料。 |

> 上面提到的結構化學習(structure learning)？在NLP的領域很常見。假設我們要區分單字的詞性，然而單字的詞性往往不能只看個體，而是需要透過整個句子的架構去判斷。所以y的輸出可能是 {pronoun/verb/noun, pronoun/verb/pronoun, noun/verb/noun, pronoun/verb...}。我們只知道類別之間會有緊密的關係。

根據協議可以分三種learning：
| 方法        | 描述  |
| ------------- |:-----|
| batch learning | 一次性餵入所有的training sample，創建model |
| online learning | sequentially學習，hypothesis是動態的不斷進步。所以跟我們的PLA和強化學習都很合得來，PLA會透過一筆一筆的錯誤去作修正... |
| active learning | 我們希望machine自己問問題，**improve hypothesis with fewer labels by asking questions strategically** (通常用於取得label成本較高的時候) |

## 來談談我們不同種類的input
| 輸入        | 描述  |
| ------------- |:-----|
| concrete feature | 具體的特徵，對ML也是最容易使用的輸入。e.g. 字跡對稱性，密度 |
| raw feature | 稍微抽象 e.g. 灰階256*256的各個數值 |
| abstract feature | 完全抽象，沒有含義 |

所以我們要把feature都轉換為比較有意義的像是concrete feature！這個過程也稱為**特徵工程(feature engineering)**。

## Training versus Testing

將Model訓練出來後我們也要了解Model是否有能力預測? Model的預測能力是多少? 這時我們就要進行Testing。

利用過去的資料我們可以使<a href="https://www.codecogs.com/eqnedit.php?latex=E_{in}\approx&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E_{in}\approx&space;0" title="E_{in}\approx 0" /></a> 代表訓練出來的Model預測Input的錯誤接近於0，但這不代表Model可以完美地預測。如果訓練出來的Model只能表現input卻不能對未知的資料進行預測的話，這個也只是一個overfitting的Model。因此我們必須訓練出可以將<a href="https://www.codecogs.com/eqnedit.php?latex=E_{out}\approx&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E_{out}\approx&space;0" title="E_{out}\approx 0" /></a>的Model(Model對預測未知資料錯誤接近於0)代表具有預測的能力。
在<a href="https://www.codecogs.com/eqnedit.php?latex=E_{in}\approx&space;E_{out}\approx&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E_{in}\approx&space;E_{out}\approx&space;0" title="E_{in}\approx E_{out}\approx 0" /></a> 這樣的基礎下，<a href="https://www.codecogs.com/eqnedit.php?latex=E_{out}\approx&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E_{out}\approx&space;0" title="E_{out}\approx 0" /></a>是機器在訓練的部分，而<a href="https://www.codecogs.com/eqnedit.php?latex=E_{in}\approx&space;E_{out}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E_{in}\approx&space;E_{out}" title="E_{in}\approx E_{out}" /></a>是測試這個Model可不可行的階段。.

#### 想要衡量模型的好壞我們必須提出以下兩個問題: Trade Off On M(Amount Of Hypothesis Set)
- 1.我們可以確定<a href="https://www.codecogs.com/eqnedit.php?latex=E_{out}\left&space;(&space;g&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E_{out}\left&space;(&space;g&space;\right&space;)" title="E_{out}\left ( g \right )" /></a>靠近<a href="https://www.codecogs.com/eqnedit.php?latex=E_{in}\left&space;(&space;g&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E_{in}\left&space;(&space;g&space;\right&space;)" title="E_{in}\left ( g \right )" /></a>嗎?
- 2.<a href="https://www.codecogs.com/eqnedit.php?latex=E_{in}\left&space;(&space;g&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E_{in}\left&space;(&space;g&space;\right&space;)" title="E_{in}\left ( g \right )" /></a>他夠小嗎?

從上面兩個問題我們可以知道M的選擇直接影響了機器學習的核心問題 -> 機器能不能學習?

#### 我們比較一下M(amount of hypothesis set)的大小所帶來的影響

| M              | 優點  |缺點  |
| ------------- |:-----|:-----|
| 少量 |會發現M帶入P[BAD]=<2*M*exp(...)公式，因為M小，P[BAD]也會變小 |因為M小，選擇太少了|
| 大量 | 因為M大，選擇很多，可以找到夠小的Ein(g) | P[BAD]增加|

從上面我們可以知道選擇是很重要的，並不是越多或越少就是越好。既然無限的M是不可行的，我們假設在有限的<a href="https://www.codecogs.com/eqnedit.php?latex=m_{\left&space;(&space;H&space;\right&space;)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m_{\left&space;(&space;H&space;\right&space;)}" title="m_{\left ( H \right )}" /></a>找M似乎是可行的。

#### 那我們該如何找到M(Amount of Hypothesis Set)

##### Over-Estimating
對我們來說<a href="https://www.codecogs.com/eqnedit.php?latex=E_{out}\left&space;(&space;g&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E_{out}\left&space;(&space;g&space;\right&space;)" title="E_{out}\left ( g \right )" /></a>與<a href="https://www.codecogs.com/eqnedit.php?latex=E_{in}\left&space;(&space;g&space;\right&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E_{in}\left&space;(&space;g&space;\right&space;)" title="E_{in}\left ( g \right )" /></a>相差太多就是不好的Hypothesis，如果我們把所有的hypothesis發生不好的機率找到並聯集起來就可以找出所有的Bad Event。假設M = ∞時，Bad Event = ∞，這樣我們尋找Bad Event就變成無意義了。
但如果我們來探討每個Hypothesis會發現並不是所有的Bad Event都是獨立的，這代表我們利用聯集來找所有的Bad Event是不盡然正確而且會造成Over-Estimating。

##### 分類Hypothesis
為了解決Over-Estimating，將Hypothesis進行分類便可以找出相似的Hypothesis。那我們開如何歸類呢? 我們假設在平面上有幾個點(Hypothesis)並思考要如何利用直線在平面上將點分開。如果有兩個點的話我們只需要一條線就可以分開並會出現四種分法[(0,0),(X,X), (0,X), (X,0)]，如果有三個點的話會出現把八種分法[(0,0,0), (X,X,X), (0,X,X), (X,0,X), (0,0,X), (X,X,0), (0,X,0), (X,0,0)]，再把共線性考慮進去的話則是少於八種。歸納以上我們可以發現分割的線會小於<a href="https://www.codecogs.com/eqnedit.php?latex=2^{N}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{N}" title="2^{N}" /></a>。我們稱這些線為有效的線(Effective Number of Lines)。

##### Dichotomy & Growth Function
Dichotomy (二分法)就是指上述在平面或是一度空間將值分為兩類的方法，我們可以知道當我們有多個Hypothesis便會出現小於<a href="https://www.codecogs.com/eqnedit.php?latex=2^{N}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{N}" title="2^{N}" /></a>個Dichotomies。
但是卻會依賴X而產生不同的Dichotomy，為了解決這個問題，利用Growth Function來找出
Growth Function (成長函數)是指在N個Hypothesis中<a href="https://www.codecogs.com/eqnedit.php?latex=\left&space;|&space;H(X_{1},X_{2},X_{3}...,X_{n})&space;\right&space;|" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\left&space;|&space;H(X_{1},X_{2},X_{3}...,X_{n})&space;\right&space;|" title="\left | H(X_{1},X_{2},X_{3}...,X_{n}) \right |" /></a>找出最大的dichotomy。
如果我們從單一dichotomy來看的話，在一度的Postive rate上可以發現:如果我們有n個點的話，<a href="https://www.codecogs.com/eqnedit.php?latex=m_{H}(N)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m_{H}(N)" title="m_{H}(N)" /></a> = N+1
如果我們用一度的Positive Intervals來探討的話就會發現<a href="https://www.codecogs.com/eqnedit.php?latex=m_{H}(N)=&space;\binom{N&plus;1}{2}&plus;1&space;=&space;\frac{1}{2}N^{2}&plus;\frac{1}{2}N&plus;1<&space;<&space;2^{N}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m_{H}(N)=&space;\binom{N&plus;1}{2}&plus;1&space;=&space;\frac{1}{2}N^{2}&plus;\frac{1}{2}N&plus;1<&space;<&space;2^{N}" title="m_{H}(N)= \binom{N+1}{2}+1 = \frac{1}{2}N^{2}+\frac{1}{2}N+1< < 2^{N}" /></a>When N is LARGE。即使在Convex Sets上最大的<a href="https://www.codecogs.com/eqnedit.php?latex=m_{H}(N)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m_{H}(N)" title="m_{H}(N)" /></a> 也只會等於 <a href="https://www.codecogs.com/eqnedit.php?latex=N^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N^{2}" title="N^{2}" /></a>。

由上我們可以知道在大多數時候使用Growth Function可以有效地降低Over-Estimating。

##### The Four Growth Function
| Growth Function|   |
| ------------- |:-----|
| Postive rays| <a href="https://www.codecogs.com/eqnedit.php?latex=m_{H}(N)=N&plus;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m_{H}(N)=N&plus;1" title="m_{H}(N)=N+1" /></a>  |
| Positive Intervals|<a href="https://www.codecogs.com/eqnedit.php?latex=m_{H}(N)=\frac{1}{2}N^{2}&plus;\frac{1}{2}N&plus;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m_{H}(N)=\frac{1}{2}N^{2}&plus;\frac{1}{2}N&plus;1" title="m_{H}(N)=\frac{1}{2}N^{2}+\frac{1}{2}N+1" /></a> |
| Convex Sets | <a href="https://www.codecogs.com/eqnedit.php?latex=m_{H}(N)=2^{N}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m_{H}(N)=2^{N}" title="m_{H}(N)=2^{N}" /></a> |
| 2D Perceptrons | <a href="https://www.codecogs.com/eqnedit.php?latex=m_{H}(N)<&space;2^{N}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m_{H}(N)<&space;2^{N}" title="m_{H}(N)< 2^{N}" /></a> in some case|

#### Break Point
在前面我們提出要用<a href="https://www.codecogs.com/eqnedit.php?latex=m_{H}(N)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?m_{H}(N)" title="m_{H}(N)" /></a>來取代M是為了讓機器能夠學習。在Four Growth Function的表裡我們可以知道Postive rays與Positive Intervals是多項式且<<<a href="https://www.codecogs.com/eqnedit.php?latex=2^{N}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?2^{N}" title="2^{N}" /></a>，這代表機器學習是可行的。Convex Sets是指數型的成長Growth Function = <a href="https://www.codecogs.com/eqnedit.php?latex=N^{2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N^{2}" title="N^{2}" /></a>。而2D Perceptrons，我們在先前的的[分類Hypothesis](#分類Hypothesis)



## Theory of Generalization
![圖](./screenshot/5.jpg)

## The VC Dimension 
![圖]( https://github.com/shinmao/ML-_-DL/blob/dev2/screenshot/6.jpg "圖片名稱")
![圖]( https://github.com/shinmao/ML-_-DL/blob/dev2/screenshot/7.jpg "圖片名稱")

#### M and dvc:

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

##### 在數量大的dvc:
- 1.No! P[BAD]增加
- 2.Yes! 很多權利

## Noise and Error

##### VC bound在有雜訊的情況下會不會work?
當然

##### 目標分佈P(y|x) (理想的mini-target + noise):
例子:P(<font color="blue">o<font>|x)=0.7, P(<font color="#f00">x<font>|x)=0.3
- ideal mini-target f(x)=o
- noise level=0.3
  
##### Goal of Learning:
預測ideal mini-target(w.r.t P(y|x))
在often-seen inputs(w.r.t P(y|x))
  
##### 兩個主要的error measure:
(1) 0/1 error:
- 直接用於對或錯
- 常用在classification

(2) squared error
- 算y~和y之間的距離
- 常用在regression

![圖](./screenshot/L8-2.jpg)
![圖](./screenshot/L8-3.jpg)
![圖](./screenshot/L8-4.jpg)
##### 結論: ideal mini-target是noise和error組成的

##### True negative, false negative有趣的例子:
- Supermarket和CIA指紋辨識的false accecpt和false reject的成本

找Ew in(h)最小值，即可得出最好的Ew in(h)
當然因為比較慢啊  

## Validation
* [Validation](./validation/README.md)

## 學習的可行性
* [學習的可行性](./Feasibility-Learning/README.md)

## linear classification
* [線性分類](linear-classification/README.md)
* [線性判別函數(linear discriminant function)](linear-classification/README.md)
* [Fisher線性判別函數](linear-classification/README.md)
* [PLA](linear-classification/README.md)
* [推薦閱讀：線性判別分析](https://ccjou.wordpress.com/2014/03/20/%E7%B7%9A%E6%80%A7%E5%88%A4%E5%88%A5%E5%88%86%E6%9E%90/)

## Linear Regression
* [線性迴歸](./linear-regression/README.md)

### Generative model v.s. Discriminative model
* [Generative model v.s. Discriminative model](./Gen-Model-and-Dis-Model/README.md)

### SVM
* [先從kernel trick是啥講起](./SVM/README.md)
* [SVM入場](./SVM/README.md)

## DL
* [什麼是Nueral Network?](./DL/README.md)
* [CNN](./DL/README.md)
* [BP (Back Propagation)](./back-propagation/README.md)
