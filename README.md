# ML-_-DL
- [ML-_-DL](#ml-_-dl)
  - [ML](#ml)
  - [ML數學概率基礎](#ml數學概率基礎)
  - [維度](#維度)
  - [學習方法：](#學習方法)
  - [來談談我們不同種類的input](#來談談我們不同種類的input)
  - [validation](#validation)
  - [學習的可行性](#學習的可行性)
  - [linear classification](#linear-classification)
    - [用linear classification解釋generative model和discriminative model](#用linear-classification解釋generative-model和discriminative-model)
  - [linear regression](#linear-regression)
  - [DL](#dl)
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
<img src="./screenshot/gaussian.png" width="50%">  
看了這張圖，有沒有清楚許多呢？  
而且lvalue必然大於等於0，然後積分也為1。所以把他當成概率密度函數也不為過！  
當然真實世界中我們不會是一元的分佈，像上面在講貝式的時候我們就有`D`維的向量，`D`維的高斯分佈大概長這樣吧(看看就好的概念)...  

<a href="https://www.codecogs.com/eqnedit.php?latex=N(x|\mu&space;,\sum&space;)=&space;\frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{(\left&space;|&space;\sum&space;\right&space;|)^\frac{1}{2}}&space;exp\left&space;\{&space;-\frac{1}{2}(x-\mu)^{T}\sum&space;(x-\mu)&space;\right&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N(x|\mu&space;,\sum&space;)=&space;\frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{(\left&space;|&space;\sum&space;\right&space;|)^\frac{1}{2}}&space;exp\left&space;\{&space;-\frac{1}{2}(x-\mu)^{T}\sum&space;(x-\mu)&space;\right&space;\}" title="N(x|\mu ,\sum )= \frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{(\left | \sum \right |)^\frac{1}{2}} exp\left \{ -\frac{1}{2}(x-\mu)^{T}\sum (x-\mu) \right \}" /></a>  

現在假定我們從D維的dataset中，做多次抽取，而每次的抽取事件也都互相獨立，別稱i.i.d. (independent and identically distributed)。那我們可以給我這些事件的聯合分佈：  

<a href="https://www.codecogs.com/eqnedit.php?latex=p(x|\mu,&space;\sigma&space;^2)=&space;\prod_{n=1}^{N}N(xn|\mu,&space;\sigma&space;^2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(x|\mu,&space;\sigma&space;^2)=&space;\prod_{n=1}^{N}N(xn|\mu,&space;\sigma&space;^2)" title="p(x|\mu, \sigma ^2)= \prod_{n=1}^{N}N(xn|\mu, \sigma ^2)" /></a>  
這其實就是gaussian的likelihood function啦！  
一樣我們的目標是maximum likelihood，實際應用中我們會考慮求likelihood的對數值更為方便。因為對數函數是單調遞增，所以我們最大化likelihood相當於在最大化對數值。  
<img src="screenshot/loggaussian.png" width="50%">  
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

## 維度
* 維度災難 (Curse of dimensionality)
* PLA 降維

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

## linear classification
分類的目標是將input分成k個離散的類型 e.g. C1, C2, ...Ck。  
模型通常是linear function。如果dataset可以被我們的model完整得分類，那這個dataset可以稱作linear separable。  

<a href="https://www.codecogs.com/eqnedit.php?latex=y(x)&space;=&space;f(w^{T}x&space;&plus;&space;w0)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y(x)&space;=&space;f(w^{T}x&space;&plus;&space;w0)" title="y(x) = f(w^{T}x + w0)" /></a>    
最簡單的情況下，模型也是由`w`和`x`的線性函數所組成的。這一類model被稱作推廣的線性模型(generalized linear model)，但跟下一個section的regression不一樣的地方是: `f()`將原本的線性函數轉換為非線性函數。  

> 線性判別函數(linear discriminant function)  

discriminant function透過input將該筆data分類  
最簡單的形式就是上面見到的`y(x) = w^Tx + w0`，`w`為權重的向量，`w0`為bias。  
如果將這個公式設為0當作我們的決策平面：若`y(x) >= 0`則分到A類，否則分到B類。那我們假設x1, x2為這個決策平面上的兩個點(兩點相連則為決策平面上的一條向量)，代入公式：  
```
y(x1) = w^Tx1 + w0 = 0
y(x2) = w^Tx2 + w0 = 0
w^T(x1 - x2) = 0
```
所以決策平面上的向量跟`w^T`作內積會得到0: 代表`w`是一條決策平面的法向量，換句話說`w`決定了我們平面的方向！  
再舉個例子：從原點到某個點x的向量 = `x'`(x向量在平面上的投影) + 垂直向量  

<a href="https://www.codecogs.com/eqnedit.php?latex=x&space;=&space;x'&space;&plus;&space;r\cdot&space;\frac{w}{|w|}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?x&space;=&space;x'&space;&plus;&space;r\cdot&space;\frac{w}{|w|}" title="x = x' + r\cdot \frac{w}{|w|}" /></a>  
我們兩邊同時乘上`w^T` 並且加上 `w0`  
代入等式`y(x) = w^Tx + w0`和`y(x') = w^Tx' + w0 = 0`，最後可以得到  

<a href="https://www.codecogs.com/eqnedit.php?latex=y(x)&space;=&space;r&space;\cdot&space;\frac{w\cdot&space;w^T}{|w|}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y(x)&space;=&space;r&space;\cdot&space;\frac{w\cdot&space;w^T}{|w|}" title="y(x) = r \cdot \frac{w\cdot w^T}{|w|}" /></a>  
而 `w * w^T = |w|^2`:  

可得  <a href="https://www.codecogs.com/eqnedit.php?latex=r&space;=&space;\frac{y(x)}{|w|}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?r&space;=&space;\frac{y(x)}{|w|}" title="r = \frac{y(x)}{|w|}" /></a>  

> 那我們的discriminant function會怎麼處理multiple class呢？  

這裡先附上一張圖：  
<img src = "./screenshot/1vrand1v1.png" width = "50%">  

左邊那張圖是**one versus rest**的處理方法，會用(k-1)個classifier切分輸入空間，缺點是會有不知道如何是好的區域。  
右邊的圖則是**one versus one**的處理方法，一個classifier用來區分任意兩個class，所以會有(k*(k - 1)/2)個classifier，然而還是會有那個神秘區域。  
  
好的！解決辦法就是引入 Class M的判別函數: `ym(x) = wm^Tx + wm0`，由m個線性函數組合而成的。  
對於某個點x，其他所有非m類的Class n，若是 `ym(x) > yn(x)`，則我們會將x分到m類。所以Cm類與Cn類的決策平面將為 `ym(x) = yn(x)`  

進一步我們可以得到  <a href="https://www.codecogs.com/eqnedit.php?latex=(wm&space;-&space;wn)^Tx&space;&plus;&space;(wm0&space;-&space;wn0)&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(wm&space;-&space;wn)^Tx&space;&plus;&space;(wm0&space;-&space;wn0)&space;=&space;0" title="(wm - wn)^Tx + (wm0 - wn0) = 0" /></a>  

> Fisher線性判別函數  

這個方法是透過投影進低維實現的，先來張圖：  
<img src = "./screenshot/fisher.png" width = "50%">  
可以看到上圖投影進`w^T`的方向，結果造成兩個class的點重疊很多，這是因為投影後會丟失很多訊息。所以在使用這個方法的時候需要慎選`w^T`。上圖如果是選水平線作為投影的目標那就能完美區分了。至於區分的標準則是為y值設置一個threshold。這也正是LDA的目標 (Linear Discriminant Analysis)，那我們要怎麼找到這個絕佳的方向呢？  

假設最簡單的binary classification: C1, C2  
m1則為C1群的平均向量，m2則為C2群的平均向量  
要看兩個的區分程度可以從 `m2 - m1` 所得：而我們要找到一個`w^T`讓他投影後得到 `w^T(m2 - m1)` 的最大值！  
fisher的思想除了讓mean的投影分得越開，還有讓同樣的class內部的方差變得越小。  
想當於這個的比值：  

<a href="https://www.codecogs.com/eqnedit.php?latex=max&space;\frac{(m2&space;-&space;m1)^2}{s1^2&space;&plus;&space;s2^2}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?max&space;\frac{(m2&space;-&space;m1)^2}{s1^2&space;&plus;&space;s2^2}" title="max \frac{(m2 - m1)^2}{s1^2 + s2^2}" /></a>  
分子m的差距越大越好，而分母則為兩個class內部的方差總和：  

<a href="https://www.codecogs.com/eqnedit.php?latex=si^2&space;=&space;\sum_{x->Ci}^{}(w^Tx&space;-&space;mi)^2" target="_blank"><img src="https://latex.codecogs.com/gif.latex?si^2&space;=&space;\sum_{x->Ci}^{}(w^Tx&space;-&space;mi)^2" title="si^2 = \sum_{x->Ci}^{}(w^Tx - mi)^2" /></a>  
略過公式的代入細節，我們可以得到：  

<a href="https://www.codecogs.com/eqnedit.php?latex=J(w)&space;=&space;\frac{w^T\cdot&space;SB\cdot&space;w}{w^T\cdot&space;SW\cdot&space;w}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?J(w)&space;=&space;\frac{w^T\cdot&space;SB\cdot&space;w}{w^T\cdot&space;SW\cdot&space;w}" title="J(w) = \frac{w^T\cdot SB\cdot w}{w^T\cdot SW\cdot w}" /></a>  
在`J(w)`最大值的時候我們可以發現：  

<a href="https://www.codecogs.com/eqnedit.php?latex=SB\cdot&space;w&space;=&space;(m1&space;-&space;m2)(m1&space;-&space;m2)^Tw" target="_blank"><img src="https://latex.codecogs.com/gif.latex?SB\cdot&space;w&space;=&space;(m1&space;-&space;m2)(m1&space;-&space;m2)^Tw" title="SB\cdot w = (m1 - m2)(m1 - m2)^Tw" /></a>  
我們關心的是方向而非大小，所以像後面的`(m1-m2)^Tw`都可以不用理他了，會發現`w`在`SW^-1(m1 - m2)`的方向上。(SW為class內協方差矩陣  
看看下面的圖會更有感覺呦：  
<img src = "./screenshot/lda.png" width = "50%">

> 透過PLA演算法(Perceptron Learning Algorithm)來分類  

<a href="https://www.codecogs.com/eqnedit.php?latex=y(x)&space;=&space;f(w^T\o&space;(x))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y(x)&space;=&space;f(w^T\o&space;(x))" title="y(x) = f(w^T\o (x))" /></a>. 
輸入向量會經由非線性轉換得到一個特徵向量：`φ(x)`，然後這個特徵向量將會用來構造一個linear model。其中`f()`作為activation function會輸出離散值：1 or -1。  

> 那麼，如何確定我們的w呢？ perceptron criterion

一樣可以由使誤差函數最小化的思想中得到，一個很直觀的方法便是看misclassification的總數，但這樣會很容易讓我們的誤差函數變得不連續(太難搞啦！)，因此我們再考慮另一個誤差函數唄！有一個誤差函數叫`perceptron criterion(感知機準則)`，這個想法其實很好懂喔: 透過上面的公式，我們已經知道model輸出為1 or -1，我們現在用變量t來取代為一個`{1, -1}`的集合。對於c1類，我們可以得到輸出值大於0，而對於c2類我們可以得到輸出值小於0。這是不是也代表著，只要分類對了：  

<a href="https://www.codecogs.com/eqnedit.php?latex=w^T\phi&space;(xn)tn&space;>&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w^T\phi&space;(xn)tn&space;>&space;0" title="w^T\phi (xn)tn > 0" /></a>  
所以如果是誤分類的，我們就要盡量讓誤差越小越好：  

<a href="https://www.codecogs.com/eqnedit.php?latex=Ep(w)&space;=&space;-\sum_{n->M}^{}w^T\phi&space;(xn)tn" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Ep(w)&space;=&space;-\sum_{n->M}^{}w^T\phi&space;(xn)tn" title="Ep(w) = -\sum_{n->M}^{}w^T\phi (xn)tn" /></a>  
這便是誤分類的data的誤差總和，`M`即為誤分類的集合。這樣我們也順利的將誤差函數的結果變為線性連續變化的了。下面是林軒田教授在上課講解的簡易`w`更新的公式：  
![](./screenshot/PLA1.png)  
`wt + 1`為修正後的權重，`t`則是代表第幾輪  
更細節的話：  

<a href="https://www.codecogs.com/eqnedit.php?latex=w^{\gamma&plus;1}&space;=&space;w^{\gamma}&space;-&space;\eta&space;\bigtriangledown&space;E_{p}(w)&space;=&space;w^\gamma&space;&plus;&space;\eta&space;\phi&space;_{n}t_{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w^{\gamma&plus;1}&space;=&space;w^{\gamma}&space;-&space;\eta&space;\bigtriangledown&space;E_{p}(w)&space;=&space;w^\gamma&space;&plus;&space;\eta&space;\phi&space;_{n}t_{n}" title="w^{\gamma+1} = w^{\gamma} - \eta \bigtriangledown E_{p}(w) = w^\gamma + \eta \phi _{n}t_{n}" /></a>  
其中`η`為learning rate，`τ`為一個整數(代表的是次數，跟上面簡化版的t一樣)，這是一個隨機梯度下降的算法。  

> 好的，我們來整理一下PLA的整個過程吧

我們不斷計算`y(x)`的值。如果分類正確，那麼`w`不變; 如果分類錯誤，我們會把向量`φ(xn)`加到`w`向量上if屬於c1類，或從中減掉`φ(xn)`if屬於c2類。  
<img src = "./screenshot/pla_process.png" width = "50%">  
我已經為大家列上圖片的順序囉！第一張圖的黑色箭頭作為初始的`w`向量，相應的黑色直線則為decision boundary。綠色的點就是我們選的第一個要糾正的點，對於紅色c1類。我們要將他的特徵向量加到我們的`w`上，因次新的decision boundary變動成第二張圖。第三張圖我們再次選了一個誤分類的點，一樣作了變化，整個過程後就能得到正確的decision boundary囉！  

> PLA怎麼停下來？ convergence theorem  

每一次修正的`wt+1`都會離目標權重`wf`更近，數學觀念上就是內積越來越大  
我們可能會懷疑內積越來越大不是因為兩線的夾角變小，而是因為向量的長度變大  
但通過證明發現，我們的mistake反而限制了我們向量長度的生長，因此長度對內積的影響並不會大到哪裡去  
而且根據公式，`兩個向量的夾角大於等於次數的根號與一個常數的product`  
因此T是有上限的 **(only if linear separable)**!

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

> 推薦閱讀  

[線性判別分析](https://ccjou.wordpress.com/2014/03/20/%E7%B7%9A%E6%80%A7%E5%88%A4%E5%88%A5%E5%88%86%E6%9E%90/)

### 用linear classification解釋generative model和discriminative model
generative model: 生成模型  
discriminative model: 判別模型  
常聽到的GAN，它是由兩層neural network所構成，一層訓練生成模型，另外一層訓練判別模型，兩層在訓練過層中互相對抗，因此而不斷更新兩個模型中的參數。這裡我們就用binary classification來做解釋。  

貝氏精神：  
之前有提過PRML非常的信仰貝氏對吧？上面的[ML數學概率基礎](#ml數學概率基礎)也提過後驗機率和生成模型之類的東東，這裡就來做個了結吧。supervised learning的目標基本上也是逼近整個dataset的機率分配。而兩個模型的不同在於他們想要逼近的機率分配不同！三種重要的機率分配：  
* 先驗分配(prior distribution)：目標變數的分配`P(Y): y -> [0, 1]`
* 聯合分配(joint distribution)：特徵向量與目標變數的分配`P(X, Y)`，也是我們一般最常蒐集到的training data
* 後驗分配(posterior distribution)：目標函數在特徵向量given時的分配`P(Y|X): y -> [0, 1]`

判別模型是透過在大量資料上估計posterior distribution的學習方法。生成模型則是估計joint distrubution`P(X, Y)`，再透過貝氏定理得出`P(X|Y)`。  

生成模型  
假設我們拿到一個x，我們想知道他屬於class 1還是class 2，就由`p(C1|x) v.s. p(C2|x)`來決勝負唄  
<a href="https://www.codecogs.com/eqnedit.php?latex=p(C1|x)&space;=&space;\frac{p(C1)p(x|C1)}{p(C1)p(x|C1)&space;&plus;&space;p(C2)p(x|C2)}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(C1|x)&space;=&space;\frac{p(C1)p(x|C1)}{p(C1)p(x|C1)&space;&plus;&space;p(C2)p(x|C2)}" title="p(C1|x) = \frac{p(C1)p(x|C1)}{p(C1)p(x|C1) + p(C2)p(x|C2)}" /></a>  
換句話說：`p(x) = p(x|C1)p(C1) + p(x|C2)p(C2)`，為何這叫作生成模型就是因為我們可以拿這個model去生成一個x。由上面推導可知我們要計算出答案還需要`p(Ci)`和`p(x|Ci)`。`p(Ci)`還很簡單，我們當然知道dataset中clss的比重。但怎麼得到`p(x|Ci)`呢？重點是我們想生成x，代表x還未存在於dataset中，我們總不能說`p(x|Ci) = 0`吧 = =  
這裡我們要應用到高斯分佈。其實已有的training sample也都是feature vector所構成的而已。所以我們可以找到那個產生sample的高斯分佈，並且生成這個未知x的機率也不會是0！  
<a href="https://www.codecogs.com/eqnedit.php?latex=\[&space;f(x)=\frac&space;1{\sqrt{2\pi&space;}\sigma&space;}\exp&space;\left\{&space;-\frac&space;12\left(&space;\frac{x-\mu&space;}%&space;\sigma&space;\right)&space;^2\right\}&space;\]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\[&space;f(x)=\frac&space;1{\sqrt{2\pi&space;}\sigma&space;}\exp&space;\left\{&space;-\frac&space;12\left(&space;\frac{x-\mu&space;}%&space;\sigma&space;\right)&space;^2\right\}&space;\]" title="\[ f(x)=\frac 1{\sqrt{2\pi }\sigma }\exp \left\{ -\frac 12\left( \frac{x-\mu }% \sigma \right) ^2\right\} \]" /></a>  
這是高斯分佈的機率密度函數，`miu`代表平均值會影響到機率分佈的最高點，`sigma`代表方差會影響到機率分佈的密度，所以我們只要找到這兩個參數即可。方法便是將用maximum likelihood: 將所有sample點代入機率密度函數找最大值，各別`miu`和`sigma`的偏微分 = 0便能得到我們高斯函數的最終參數！這樣一樣，我們也拿到`p(x|Ci)`囉！  
![](screenshot/classify-with-generative-model.png)  

> 其實不同的class可以share同一個covariance matrix。畢竟參數如果太多很容易overfitting，所以我們可以強迫兩個class的gaussian function使用同樣的covariance參數。其實新的covariance參數求法也非常直觀，假設原本是sigma1和sigma2，新sigma = p(C1)*sigma1 + p(C2)*sigma2

> 為什麼要用guassian distribution呢？誰說的，當然可以自己選啊。如果選參數較少的distribution function，bias較大/variance較小，反之則bias較小/variance較大

上面的generative model其實可以得到**驚人**的結論。我們現在可以計算出`p(C1|x)`的值了對吧，經過複查布拉布拉布拉的計算 + 共用同一個sigma，我們可以得出最後的:  
![](screenshot/generative%20result.png)  
我們先看右邊的第二條紅線，完全沒有x的變數，代表他其實就是個scalar，那麼我們可以認為他是常數，用一個`b`來做替換。而前面的`miu1 - miu2`我們用w這個vector來作替換。咦？是不是很眼熟？沒錯，這就是我們的`w^T + b`囉！這也能解釋為什麼我們將sigma共用之後得到的decision boundary會是linear的。  
上面這件事稱作logistic regression，這邊比較一下logistic regression和linear regression:  
![](screenshot/logisticvslinear.png)  
先比較function，logistic regression經由sigmoid function會壓縮到0, 1之間，而linear regression會輸出任意值。綠色框框的部分是我們評估logistic funcion的方法，一樣取個自然對數 + 負值，我們的目標就是使這個`L(w)`最小。用gradient descent對w取偏微分，可以得到:  
<a href="https://www.codecogs.com/eqnedit.php?latex=\sum_{n&space;=&space;1}^{N}(y_{n}&space;-&space;t_{n})X_{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sum_{n&space;=&space;1}^{N}(y_{n}&space;-&space;t_{n})X_{n}" title="\sum_{n = 1}^{N}(y_{n} - t_{n})X_{n}" /></a>  
這是不是也很熟悉呀？沒錯，這也是w的梯度，會隨著輸出值與目標值的差距調整梯度。  

現在回到generative model v.s. discriminative model  
他們兩個的目標函數是一樣的：  

<a href="https://www.codecogs.com/eqnedit.php?latex=p(Ci|x)&space;=&space;\sigma&space;(wx&space;&plus;&space;b)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(Ci|x)&space;=&space;\sigma&space;(wx&space;&plus;&space;b)" title="p(Ci|x) = \sigma (wx + b)" /></a>  
**兩個都需要找出w和b**。generative model需要找出c1的平均值，c2的平均值，和兩個共用的covariance，進一步算出w和b。而discriminative model則直接找出w和b(方法就是上面的cross entropy取gradient descent)。但兩個找出的參數會有所不同，因為假設的不一樣，e.g. generative model中我們就假設了guassian distribution。**通常discriminative model找出的會比generative model還要精準！**  

為什麼discriminative mode常常比generative model還要精準呢？因為generative model很會腦補。基本上他使用的是naive Bayes，代表他假設了所有feature dimension之間都是互相獨立的。所以他很容易忽略feature之間的correlation。在這方面。discriminative model則非常老實的看data說話。不過generative model在兩個情況下派得上用場：1. training sample很少時，我們需要自己腦補一些假設，2. dataset很多noise時，看data說話很容易被誤導。  

> 不過logistic regression有很大的限制性。data非linear separable時我們可以進行feature transformation。這甚至還跟後來的neural network息息相關...

> 推薦閱讀

[邏輯斯回歸](https://ccjou.wordpress.com/2014/03/26/%E9%82%8F%E8%BC%AF%E6%96%AF%E5%9B%9E%E6%AD%B8/)

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

## DL
`Perceptron has limitation` -> `multi-layer perceptron` -> `backpropagation` -> `1 hidden layer is good enough` -> `Acceleration with GPU` -> `DL in speech recognition and image`  

不同的Neuron合併或串接在一起組成Neural network  
每一個neuron都是由logistic regression組成  
不同的串接方式會有不同的網路架構  
回去看看我們的logistic regression有哪些參數：`weight`，`bias`  
因此neural network的參數便是所有neuron的`weight`和`bias`  

我們可以把neural network視作一個function，input和output都是vector  
如果我們只有neural network的連接方式，但還沒有參數，此時的neural network則是function set  
<img src = "./screenshot/NN.png" width = "50%">  
上面是一種fully connect feedforward network，前一個layer的所有output便是後一個layer的所有input  
除了最一開始的input layer和最後的output layer，中間的都是hidden layer(deep便是多層hidden layers的意思)，每一層layer由數多個neuron所組成  
<img src = "./screenshot/NN-structure.png" width = "50%">  

別看neuron這麼多很多output會很麻煩  
既然是vector我們當然可以把一層的input/output/weight/bias換成matrix  
因此neural network可以視為一層一層的matrix operation和sigmoid function(聽說現在都不是丟sigmoid啦...)  
常聽到的GPU加速也是在幫我們做矩陣運算之類的事情  

> 對於output layer而言  

我們當然知道他不是直接從input layer拿那些input當feature  
而是經由多層hidden layers作轉換 - 這個過程就好像 feature engineering  
這些features送到output layer之後，輸出最後的y1, y2, ...ym  
這個output layer可以視為multi-class classifier  
neural network的output便是output layer使用了softmax，那個output代表了各個結果的probability distribution  
e.g. 我們把一張手寫數字的筆跡送進這個neural network  
輸入便是整張圖16*16的pixel，每個pixel只有兩種輸出: 0代表白，1代表黑，總共256維度的vector  
輸出則是10維度的vector，代表從0, 1~9的機率。假如2的機率佔了足足`0.7`，那這張圖代表2的機率就非常高了。  
我們知道了input和output的維度，剩下就是決定中間的structure了  
**這裡很重要：**  
input和output的dimensions很容易得知，剩下就是network的架構了  
**之前在logistic regression或linear regression我們都無需設計model的架構，但對於neural network而言，中間要有幾層hidden layers，每層layer要有多少neuron都是要自己設計的，他也是決定function set的長相**。  

> 怎麼設計network structure呢？

我們來比較看看語音/影像辨識和NLP  
network structure v.s. feature engineering  
上面有提到中間的hidden layer就好像在作feature transformation  
**但其實deep learning根本不需要feature extraction的動作**  
我們都知道DL在語音和影像辨識的部分風生水起  
透過DL，我們完全不需要抽取部分pixel，而是把整個pixel丟進去硬幹！  
所以問題變成**選取重要feature好 還是 設計network架構好？**  
:wink: 語音/影像辨識  
先給個結論，這部分直接作network structure容易太多了  
因為這個東西離人類實在無法理解，機器覺得重要的向量我們卻看不出來  
因此嘗試各種network structure讓機器自己找出好的feature才是正解。  
:wink: NLP  
不過DL在NLP上就不怎麼U秀了  
因為人類在語言這方面處理以及理解的能力比機器還要好太多了  
因此DL給NLP的進度和優勢就不怎麼顯著了  

> 如何判定好的function？

把我們的target丟到output會得到一組y值(也就是答案)  
把我們的input丟進network到最後經由softmax也會輸出一組y值  
然後跟之前在discriminative model提到的方法一樣，取兩個的cross entroy，當然會有多個cross entropy，所有cross entropy總和起來便是total loss  
我們想個辦法調整參數來讓total loss得到最小值，使用的方法便是gradient descent囉！  
gradient descent就算在DL裡面還是微分求導數之類的，本質不會變，只不過function變得很複雜  
我們針對每個weight，每個bias下去作偏微分，當然這實在是太麻煩了...  
backpropagation（反向傳播）算是一個比較有效的微分計算方式  
現在大多用tensorflow和pytorch之類的toolkit幫我們計算了  

> backpropagation

在neural network裡面作gradient descent最麻煩的地方在於上百萬個`w`和`b`參數  
backpropagation是一種比較有效率的gradient descent方法  
核心是chain rule：  
<img src="screenshot/chainrule.png" width="50%">  
還記得我們上面提過的關於NN的loss function嘛？  
<a href="https://www.codecogs.com/eqnedit.php?latex=L(\theta&space;)&space;=&space;\sum_{n=1}^{N}C^{n}(\theta&space;)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?L(\theta&space;)&space;=&space;\sum_{n=1}^{N}C^{n}(\theta&space;)" title="L(\theta ) = \sum_{n=1}^{N}C^{n}(\theta )" /></a>  
`l^n`便是其中一筆的cross entropy，全部加總便是total loss  
我們也可以把這個微分放到sigma裡面去，現在我們只focus在一筆data(l)就好  
<img src="screenshot/bkp.png" width="50%">  
單筆的loss function對w作偏微分，我們用**forward pass**和**backward pass**來解決計算的麻煩  
根據chain rule的case 1，我們可以把`loss function對w作的微分`換成`z對w的微分`和`l對z的微分`(看上圖左下)  
現在分開來計算  
`z對w的微分`簡直不能再更簡單了  
微分當然是直取該參數的係數  
這就叫做foward pass，至於原因等下就會明白了...  

現在來到`l對z的微分`了  
<img src="screenshot/bkward-pass.png" width="50%">  
假定經過我們activation function輸出的output為`a`  
作為下一個neuron的input  
`a`同樣也會乘上`w3/w4`以及加上某些`b`得到輸出`z'/z"`  
還記得我們想求`l對z的微分吧`？一樣畫葫蘆：他也可以換成`a對z的微分`和`l對a的微分`  
`a對z的微分`也就是activation function的微分  
而`l對a的微分`根據chain rule的case 2可以換成上圖右下角的樣子  
看看上圖，前面不正是`a`變成output的係數(`w3`, `w4`)嘛  
接下來來解決最後的問題：`l對z'和z"的微分`該怎麼算呢？  
<img src="./screenshot/other-perspective.png" width="50%">  
來個反向思考：把`l對z'和z"的微分`當成input，`w3`和`w4`分別為他們的weight，相加之後經由activation function的微分就可以得到我們想要的`l對z的微分`囉！  
而這個activation function的微分因為在forward pass裡就已經決定了，在這個**反向的neuron**裡就是個常數  

<img src="./screenshot/bkpp.png" width="50%">  
從output layer算回去就發現能一路算出`l對z的微分`囉！  
其實backward pass相當於建造一個新的neural network  
只是原本的activation function變成了一個常數！  

最後用一張圖再總結一次  
<img src="./screenshot/bkp-summary.png" width="50%">  

> Why deep?

這裡的問題其實是 why deep? not fat?  
根據universality theorem，一層hidden layer足以表達整個function set  
那我們為何還需要deep呢？  
那DNN豈不是變成FNN了？

## Refer
* 林軒田 - 機器學習基石
* 林軒田 - 機器學習技法
* Bishop - PRML
