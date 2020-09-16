# ML-_-DL
- [ML-_-DL](#ml-_-dl)
  - [ML](#ml)
  - [ML數學概率基礎](#ml數學概率基礎)
  - [Curse of Dimensionality](#curse-of-dimensionality)
  - [學習方法：](#學習方法)
  - [來談談我們不同種類的input](#來談談我們不同種類的input)
  - [validation](#validation)
  - [學習的可行性](#學習的可行性)
  - [How to answer yes or no?](#how-to-answer-yes-or-no)
  - [linear regression](#linear-regression)
  - [Refer](#refer)

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
:star: Bayesian probabilities  
一般的事件我們可以用高中數學統計出概率。啊如果是不確定性的事件呢？不確定性的事件就是：這些事件無法重複多次，也就是我們有生之年可能沒法統計出他的概率。我們希望用一個更通用的方式來定量化這些事件，就要用到 Bayesian probabilities。  
先複習個貝式定理：  

<a href="https://www.codecogs.com/eqnedit.php?latex=P(A|B)&space;=&space;\frac{P(B|A)P(A)}{P(B)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(A|B)&space;=&space;\frac{P(B|A)P(A)}{P(B)}" title="P(A|B) = \frac{P(B|A)P(A)}{P(B)}" /></a>  
說到我們ML中的不確定性，可以說是model的參數囉，這裡就先給個`w`！我們在觀察到dataset以前，會先對`w`做個假設，也就是prior prob：`p(w)`。觀察完dataset之後的conditional prob可以用`p(D|w)`來表示。Bayesian讓我們可以通過一個形式表示這個不確定性(也就是這裡的posterior prob)：  

<a href="https://www.codecogs.com/eqnedit.php?latex=P(w|D)&space;=&space;\frac{P(D|w)P(w)}{P(D)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?P(w|D)&space;=&space;\frac{P(D|w)P(w)}{P(D)}" title="P(w|D) = \frac{P(D|w)P(w)}{P(D)}" /></a>  
細部探討一下上面這個formula:   
`P(D|w)`可以當成我們model參數`w`的function，別稱**likelihood function**。他表示在不同參數下，該dataset出現的概率。  
`P(D)`則是確保合理的概率密度，積分為1  
<a href="https://www.codecogs.com/eqnedit.php?latex=p(D)&space;=&space;\int&space;p(D&space;|&space;w)p(w)&space;dw" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(D)&space;=&space;\int&space;p(D&space;|&space;w)p(w)&space;dw" title="p(D) = \int p(D | w)p(w)  dw" /></a>  

對於likelihood function，我們的目標是使其最大化(有沒有很熟悉maximum likelihood呀)，`w`應該使我們的likelihood funciton達到最大值(出現同樣dataset的機率越高)。而likelihood的負對數則是所謂的**error function**，根據單調成長的特性，我們讓maximum likelihood等同於minimize error function！  

:star: Gaussian distribution  
一種連續變量的正態分佈。對於一元變量x，我們將高斯分佈定義為：  

<a href="https://www.codecogs.com/eqnedit.php?latex=N(x|\mu&space;,\sigma&space;^2)=&space;\frac{1}{\sqrt{2\pi&space;\sigma^2}}&space;exp\left&space;\{&space;-\frac{1}{2\sigma^2}(x-\mu)^{2}&space;\right&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N(x|\mu&space;,\sigma&space;^2)=&space;\frac{1}{\sqrt{2\pi&space;\sigma^2}}&space;exp\left&space;\{&space;-\frac{1}{2\sigma^2}(x-\mu)^{2}&space;\right&space;\}" title="N(x|\mu ,\sigma ^2)= \frac{1}{\sqrt{2\pi \sigma^2}} exp\left \{ -\frac{1}{2\sigma^2}(x-\mu)^{2} \right \}" /></a>  
`μ`: 平均值  
`σ^2`: 變異數，也就是標準差的平方  
`1/(σ^2)`: 精準度  
<img src="./screenshot/gaussian.png" style="zoom:50%">  
看了這張圖，有沒有清楚許多呢？  
而且lvalue必然大於等於0，然後積分也為1。所以把他當成概率密度函數也不為過！  
當然真實世界中我們不會是一元的分佈，像上面在講貝式的時候我們就有`D`維的向量，`D`維的高斯分佈大概長這樣吧(看看就好的概念)...  

<a href="https://www.codecogs.com/eqnedit.php?latex=N(x|\mu&space;,\sum&space;)=&space;\frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{(\left&space;|&space;\sum&space;\right&space;|)^\frac{1}{2}}&space;exp\left&space;\{&space;-\frac{1}{2}(x-\mu)^{T}\sum&space;(x-\mu)&space;\right&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N(x|\mu&space;,\sum&space;)=&space;\frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{(\left&space;|&space;\sum&space;\right&space;|)^\frac{1}{2}}&space;exp\left&space;\{&space;-\frac{1}{2}(x-\mu)^{T}\sum&space;(x-\mu)&space;\right&space;\}" title="N(x|\mu ,\sum )= \frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{(\left | \sum \right |)^\frac{1}{2}} exp\left \{ -\frac{1}{2}(x-\mu)^{T}\sum (x-\mu) \right \}" /></a>  

現在假定我們從D維的dataset中，做多次抽取，而每次的抽取事件也都互相獨立，別稱i.i.d. (independent and identically distributed)。那我們可以給我這些事件的聯合分佈：  

<a href="https://www.codecogs.com/eqnedit.php?latex=p(x|\mu,&space;\sigma&space;^2)=&space;\prod_{n=1}^{N}N(xn|\mu,&space;\sigma&space;^2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(x|\mu,&space;\sigma&space;^2)=&space;\prod_{n=1}^{N}N(xn|\mu,&space;\sigma&space;^2)" title="p(x|\mu, \sigma ^2)= \prod_{n=1}^{N}N(xn|\mu, \sigma ^2)" /></a>  
這其實就是gaussian的likelihood function啦！  
一樣我們的目標是maximum likelihood，實際應用中我們會考慮求likelihood的對數值更為方便。因為對數函數是單調遞增，所以我們最大化likelihood相當於在最大化對數值。  
<img src="screenshot/loggaussian.png" style="zoom:30%">  
下面兩個便是我們的sample mean和sample variance。細部的數學證明我就不在這裡耗篇幅囉...  

當我們找到maximum likelihood的解時，會發現居然低估了variance。這也是我們maximum likelihood的局限性。這種問題叫作**bias**，跟polynomial curve fitting中會遇到的overfitting有關！當我們點N的數量繼續上升的時候，那bias的現象會越來越不嚴重。  

> 這邊小提醒一下，maximum likelihood的bias問題便是我們poly curve fitting中overfittin問題的核心...

:star: Decision Theory  
決策論是我們在面對不確定性時做出的最優策略  
舉個🌰，我們想給病人拍x光片判斷他有沒有得癌症。  
Training data: 假設輸入向量是x，為x光片的pixel值。輸出變量為t = 0: C1為患有癌症，t = 1: C2為不患有癌症。  
Inference: 這個問題就變成一個聯合概率分佈的問題：`p(x, Ck)`。  
Inference step: 評估`p(x, Ck)`。  
Decision step: 從x推斷Ck讓error最小化  

<a href="https://www.codecogs.com/eqnedit.php?latex=p(Ck|x)&space;=&space;\frac{p(x|Ck)p(Ck)}{p(x)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(Ck|x)&space;=&space;\frac{p(x|Ck)p(Ck)}{p(x)}" title="p(Ck|x) = \frac{p(x|Ck)p(Ck)}{p(x)}" /></a>  
所謂的error就是將x預測成錯誤的class。換句話說，我們想讓posterior prob(`p(Ck|x)`)最大。  

把x區分到對的類別會將輸入空間切成不同的區域，也就是decision region。而邊界就叫decision boundary，可以為不連續。  

:star: minimize expected loss  

<a href="https://www.codecogs.com/eqnedit.php?latex=\begin{bmatrix}&space;0&space;&&space;1000\\&space;1&space;&&space;0&space;\end{bmatrix}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\begin{bmatrix}&space;0&space;&&space;1000\\&space;1&space;&&space;0&space;\end{bmatrix}" title="\begin{bmatrix} 0 & 1000\\ 1 & 0 \end{bmatrix}" /></a>  
此為loss matrix: `[0][x]`和`[x][0]`為癌症，`[1][x]`和`[x][1]`為健康  
繼續上面癌症的🌰，結合這個loss matrix。我們判斷對都還沒事，loss為0，重點是在判斷錯的損失。把正常的人判斷為癌症，那這損失為1。如果把癌症的人判斷為正常，這損失可就嚴重了，人都會死了！  
我們的最優解是讓這loss function value最小，也就是讓平均loss最小。我們用聯合概率分佈來表示loss function的平均值：  

<a href="https://www.codecogs.com/eqnedit.php?latex=E[L]&space;=&space;\sum_{k}^{}\sum_{j}^{}\int_{Rj}^{}Lkjp(x,&space;Ck)dx" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E[L]&space;=&space;\sum_{k}^{}\sum_{j}^{}\int_{Rj}^{}Lkjp(x,&space;Ck)dx" title="E[L] = \sum_{k}^{}\sum_{j}^{}\int_{Rj}^{}Lkjp(x, Ck)dx" /></a>  
請服用`p(x, Ck) = p(Ck|x)p(x)`:  
我們只要最小化  

<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{k}^{}Lkjp(Ck|x)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{k}^{}Lkjp(Ck|x)" title="\sum_{k}^{}Lkjp(Ck|x)" /></a>  
一旦我們知道了posterior prob的值，這就很簡單囉！  

決定好decision step後，還有一種情況是reject option。癌症的🌰中，有時候會有很難判斷的x光片子要交給專家，不適合用我們自動化的decision step。這時候就會跑出一個threshold，怎樣難度的片子我們才會交給專家。

> 接下來為大家總結一下解決decision problem的三種方法喔...

-> generative model  
在inference stage，對於每個`Ck`，先獨立計算出`p(x|Ck)`並推斷出`p(Ck)`。再使用貝式算出posterior prob：  

<a href="https://www.codecogs.com/eqnedit.php?latex=p(Ck|x)&space;=&space;\frac{p(x|Ck)p(Ck)}{p(x)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(Ck|x)&space;=&space;\frac{p(x|Ck)p(Ck)}{p(x)}" title="p(Ck|x) = \frac{p(x|Ck)p(Ck)}{p(x)}" /></a>  
也就是上面的`p(Ck|x)`。一如往常，我們可以將貝式中的分母用分子中的項表示：  

<a href="https://www.codecogs.com/eqnedit.php?latex=p(x)&space;=&space;\sum_{k}^{}p(x|Ck)p(Ck)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(x)&space;=&space;\sum_{k}^{}p(x|Ck)p(Ck)" title="p(x) = \sum_{k}^{}p(x|Ck)p(Ck)" /></a>  
然後用decision step決定哪個class。  

-> disciminative model  
則是對posterior prob直接建立model。  

-> 還有一種是找到一種function能直接將input map到class label的。這就跟probability八竿子打不著了。

## Curse of Dimensionality
真實世界中，為了完美得分類，我們需要取很多樣的特徵。然而，分類器的效能在過了最適當的特徵數後，是會隨特徵數(也就維
度)下降的！  
為什麼呢？實際上dataset是由低維空間所產生的，投射到高維之後點跟點之間的距離會變長，假設我們打算用點跟點之間的距離來作區分(就像knn)，那這就麻煩了...因為單位空間內的樣本數量變少了。也就是說隨著維度提高，所需的樣本數會以指數上升(哇咧，甚至到最後可能覆蓋了整個樣本空間，這樣我們的model怎麼可能會work呀。  
換句話說，距離度量型的model在越高維的空間中就越派不上用場。還好PRML安慰我們說，這並不會阻止我們尋找高維空間中的分類：高維數據中常常存在冗餘的部分，因此後面出現了降維的方法！

## 學習方法：
| 方法        | 有沒有label | 描述  |
| ------------- |:-------------:|:-----|
| supervised learning | 全都有 | 二元分類，多元分類，回歸。按照輸出空間的話，還可以在區分出結構化學習(細節放在下面)。 |
| unsupervised learning | 沒有 | clustering |
| semi- | 有些有 |  |
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

## 學習的可行性
上面我們探討了許多種學習方法  
可是你有沒有想過: 學習真的可行嗎?  
的確，如果我們不加以限制條件，你永遠有理由可以說**我這個學習方法，回答了錯的答案！**  
就算我們在training set上都得到了完美的結果，但誰知道在這set之外的數據，我們的model能否一樣完美呢？  
其實這很正常，我們想在training set外還能得到完美的結果本就是不可能的，畢竟**No free lunch**嘛！  
No free lunch定理是說：沒有一種model能在任何情況下都表現預測得很完美！所以我們說這個model比那個model好，也只是針對特定的條件下去做比較的。  
> 啊講到這裡是想跟我說，預測本來就是不可能ㄇ？那這樣還要learn啥？

根據我們的Hoeffding's inequality(霍夫丁不等式)，我們的誤差值是有一個上限的。儘管我們不知道真正的答案是多少，sample的數量越多，誤差就會越少！
> 總結一下：如果樣本數夠大，樣本中h(x) != f(x)的機率可以推導出整個抽樣空間中h(x) != f(x)的機率。兩者的機率是PAC，所以如果前面機率是小的，那後面也是小的。

我們再引入`Ein(h)`和`Eout(h)`的觀念到Hoeffding's inequality中，`Ein(h)`是指training sample上答案錯的機率，而`Eout(h)`是指整個數據上答案錯的機率。不等式表明了`Ein(h)`是很接近`Eout(h)`的 (PAC的觀念)，可是這在兩error都很大的情況下也有可能成立。所以我們要選好model讓`Ein(h)`是小的，`Eout(h)`也才會是小的！  

以上hoeffding的誤差上限都是根據一個hypothesis的說法，**但如果是很多hypothesis呢**？  
![](screenshot/multi-hoeffding.png)  
hoeffding不等式右邊就會乘上`M`(hypothesis的個數)然後作為union bound。問題其實很好處理，如果`M`的個數有限，那麼還是有個上限在！

> 好！所以接下來我們要處理的就是這個M的問題。這個M的問題也是不小...

如果M很小的話，看起來我們的bound會比較低，但這也代表我們hypothesis的選擇比較少！  
反過來的話，hypothesis選擇多，我們的bound又變大了！  
我們得拿個東西來取代 M  

我們回到上面的union bound看看，我們把每個hypothesis的bad event假設為互相獨立，所以把全部的機率加起來。但是真的有互相獨立嗎？如果兩個hypothesis很相似，那他們的bad event理所當然會重疊吧！所以我們可能高估了這個bound...  

我想把hypothesis的數量減少，所以我把相似的hypothesis都分到同一群。分類的方法由input的結果來區分，假設在二D空間中的一個點，我們可以把它區分為 o 類，也可以是 x 類。那其實也代表，我們所有的hypothesis只有分兩種: 1. 把這個點區分為 o 類的，2. 把這個點區分為 x 類的。先不要急著說可以把hypothesis定義為`2^N`類。假設三點共線，那我們怎麼樣都**沒辦法分到八類**的，到這裡其實我們就證明了: 就算有無限多的hypothesis，我們還是能夠學習的！  

> 只要我們能夠保證我們自己的 m << 2^N，也就是右邊趨近於0，那麼 M 再怎麼大也只是個笑話！

我們現在可以把原本的hypothesis set換成**dichotomy**，他是那些能夠把空間中的點完美分類的線的集合，size上限為`2^N`。把每個hypothesis依賴於sample數的變化搞成一個growth function的話，那我們會希望growth function會是polynomial的，而非exponential的。最小需要多少sample才能讓growth function小於exponential就是我們要找的break point。

## How to answer yes or no?
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

## linear regression
先來上個linear regression用的model  
<a href="https://www.codecogs.com/eqnedit.php?latex=y(x,&space;w)&space;=&space;w0&space;&plus;&space;w1x&space;&plus;&space;w2x^2&space;&plus;&space;w3x^3&space;&plus;&space;...&space;wMx^M&space;=&space;\sum_{j=0}^{M}wjx^j" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y(x,&space;w)&space;=&space;w0&space;&plus;&space;w1x&space;&plus;&space;w2x^2&space;&plus;&space;w3x^3&space;&plus;&space;...&space;wMx^M&space;=&space;\sum_{j=0}^{M}wjx^j" title="y(x, w) = w0 + w1x + w2x^2 + w3x^3 + ... wMx^M = \sum_{j=0}^{M}wjx^j" /></a>  
講到regression，我們最常用的就是polynomial curve fitting！  
等等，為什麼我要用polynomial fitting呢？這跟taylor展開式有關。任意一個函數可以表示成x的次方和，也就是可以放到(1, x, x^2, x^3,...)所張開的空間。  

假設我們想要擬和一個sin的週期函數，怎麼樣的曲線函數(polynomial curve fitting)會最接近呢？我們可以用root-mean-square error來評估這個效果。  
下面則是方差，我們的函數值是`y(...)`。在regression中我們的目標是讓方差和最小化  
<a href="https://www.codecogs.com/eqnedit.php?latex=E(w)&space;=&space;\frac{1}{2}\sum_{n=1}^{N}\left&space;\{&space;y(Xn,&space;w)&space;-&space;tn&space;\right&space;\}^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E(w)&space;=&space;\frac{1}{2}\sum_{n=1}^{N}\left&space;\{&space;y(Xn,&space;w)&space;-&space;tn&space;\right&space;\}^2" title="E(w) = \frac{1}{2}\sum_{n=1}^{N}\left \{ y(Xn, w) - tn \right \}^2" /></a>  

當數據量不夠時，也可以用regularization避開overfit的問題  
<a href="https://www.codecogs.com/eqnedit.php?latex=E(w)&space;=&space;\frac{1}{2}\sum_{n=1}^{N}\left&space;\{&space;y(Xn,&space;w)&space;-&space;tn&space;\right&space;\}^2&space;&plus;&space;\frac{}{2}\left&space;\|&space;w&space;\right&space;\|^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?E(w)&space;=&space;\frac{1}{2}\sum_{n=1}^{N}\left&space;\{&space;y(Xn,&space;w)&space;-&space;tn&space;\right&space;\}^2&space;&plus;&space;\frac{l}{2}\left&space;\|&space;w&space;\right&space;\|^2" title="E(w) = \frac{1}{2}\sum_{n=1}^{N}\left \{ y(Xn, w) - tn \right \}^2 + \frac{}{2}\left \| w \right \|^2" /></a>  
> 這邊小小介紹一下regularization幹了什麼好事？上面公式中l會控制model的複雜程度，也就是間接決定overfit的程度

看完上面的[ml數學概率基礎](##ml數學概率基礎)，我們可以來重新審視polynomial curve overfitting的問題。  
這裡有N個輸入組成的dataset：`(x1, x2, ... xN)^T`  
和目標值組成的dataset：`(t1, t2, ... tN)^T`  
我們可以用概率分佈來表示目標值的不確定性：  

<a href="https://www.codecogs.com/eqnedit.php?latex=p(t&space;|&space;x,&space;w,&space;\beta&space;)=&space;N(t&space;|&space;y(x,&space;w,&space;\beta^{-1}))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(t&space;|&space;x,&space;w,&space;\beta&space;)=&space;N(t&space;|&space;y(x,&space;w,&space;\beta^{-1}))" title="p(t | x, w, \beta )= N(t | y(x, w, \beta^{-1}))" /></a>  
我們現在通過訓練這組dataset來決定`w`和`β`的值，假設是從上面的分佈取得的，那我們可以得到:  

<a href="https://www.codecogs.com/eqnedit.php?latex=p(t&space;|&space;x,&space;w,&space;\beta&space;)=&space;\prod_{n=1}^{N}N(t&space;|&space;y(x,&space;w,&space;\beta^{-1}))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(t&space;|&space;x,&space;w,&space;\beta&space;)=&space;\prod_{n=1}^{N}N(t&space;|&space;y(x,&space;w,&space;\beta^{-1}))" title="p(t | x, w, \beta )= \prod_{n=1}^{N}N(t | y(x, w, \beta^{-1}))" /></a>  
來吧再一次！maximum log-likelihood function，但這次我們等價得最小化square error function。到這裡我們已經有`wML`和`βML`的答案了。代入上面的概率分佈公式，我們可以預測x囉。  

Maximum Posterior(MAP approach): Given dataset, 我們通過尋找最有可能的`w`(也就是我們的max posterior)來確定`w`。這等同於最小化正則化的square error function。


## Refer
* 林軒田 - 機器學習基石
* 林軒田 - 機器學習技法
* PRML
