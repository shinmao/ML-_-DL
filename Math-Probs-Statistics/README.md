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
<img src="../screenshot/gaussian.png" width="50%">  
看了這張圖，有沒有清楚許多呢？  
而且lvalue必然大於等於0，然後積分也為1。所以把他當成概率密度函數也不為過！  
當然真實世界中我們不會是一元的分佈，像上面在講貝式的時候我們就有`D`維的向量，`D`維的高斯分佈大概長這樣吧(看看就好的概念)...  

<a href="https://www.codecogs.com/eqnedit.php?latex=N(x|\mu&space;,\sum&space;)=&space;\frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{(\left&space;|&space;\sum&space;\right&space;|)^\frac{1}{2}}&space;exp\left&space;\{&space;-\frac{1}{2}(x-\mu)^{T}\sum&space;(x-\mu)&space;\right&space;\}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?N(x|\mu&space;,\sum&space;)=&space;\frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{(\left&space;|&space;\sum&space;\right&space;|)^\frac{1}{2}}&space;exp\left&space;\{&space;-\frac{1}{2}(x-\mu)^{T}\sum&space;(x-\mu)&space;\right&space;\}" title="N(x|\mu ,\sum )= \frac{1}{(2\pi)^{\frac{D}{2}}}\frac{1}{(\left | \sum \right |)^\frac{1}{2}} exp\left \{ -\frac{1}{2}(x-\mu)^{T}\sum (x-\mu) \right \}" /></a>  

現在假定我們從D維的dataset中，做多次抽取，而每次的抽取事件也都互相獨立，別稱i.i.d. (independent and identically distributed)。那我們可以給我這些事件的聯合分佈：  

<a href="https://www.codecogs.com/eqnedit.php?latex=p(x|\mu,&space;\sigma&space;^2)=&space;\prod_{n=1}^{N}N(xn|\mu,&space;\sigma&space;^2)" target="_blank"><img src="https://latex.codecogs.com/gif.latex?p(x|\mu,&space;\sigma&space;^2)=&space;\prod_{n=1}^{N}N(xn|\mu,&space;\sigma&space;^2)" title="p(x|\mu, \sigma ^2)= \prod_{n=1}^{N}N(xn|\mu, \sigma ^2)" /></a>  
這其實就是gaussian的likelihood function啦！  
一樣我們的目標是maximum likelihood，實際應用中我們會考慮求likelihood的對數值更為方便。因為對數函數是單調遞增，所以我們最大化likelihood相當於在最大化對數值。  
<img src="../screenshot/loggaussian.png" width="50%">  
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