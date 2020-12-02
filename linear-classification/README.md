## Linear Classification
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
<img src = "../screenshot/1vrand1v1.png" width = "50%">  

左邊那張圖是**one versus rest**的處理方法，會用(k-1)個classifier切分輸入空間，缺點是會有不知道如何是好的區域。  
右邊的圖則是**one versus one**的處理方法，一個classifier用來區分任意兩個class，所以會有(k*(k - 1)/2)個classifier，然而還是會有那個神秘區域。  
  
好的！解決辦法就是引入 Class M的判別函數: `ym(x) = wm^Tx + wm0`，由m個線性函數組合而成的。  
對於某個點x，其他所有非m類的Class n，若是 `ym(x) > yn(x)`，則我們會將x分到m類。所以Cm類與Cn類的決策平面將為 `ym(x) = yn(x)`  

進一步我們可以得到  <a href="https://www.codecogs.com/eqnedit.php?latex=(wm&space;-&space;wn)^Tx&space;&plus;&space;(wm0&space;-&space;wn0)&space;=&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?(wm&space;-&space;wn)^Tx&space;&plus;&space;(wm0&space;-&space;wn0)&space;=&space;0" title="(wm - wn)^Tx + (wm0 - wn0) = 0" /></a>  

> Fisher線性判別函數  

這個方法是透過投影進低維實現的，先來張圖：  
<img src = "../screenshot/fisher.png" width = "50%">  
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
<img src = "../screenshot/lda.png" width = "50%">

> 透過PLA演算法(Perceptron Learning Algorithm)來分類  

<a href="https://www.codecogs.com/eqnedit.php?latex=y(x)&space;=&space;f(w^T\o&space;(x))" target="_blank"><img src="https://latex.codecogs.com/gif.latex?y(x)&space;=&space;f(w^T\o&space;(x))" title="y(x) = f(w^T\o (x))" /></a>. 
輸入向量會經由非線性轉換得到一個特徵向量：`φ(x)`，然後這個特徵向量將會用來構造一個linear model。其中`f()`作為activation function會輸出離散值：1 or -1。  

> 那麼，如何確定我們的w呢？ perceptron criterion

一樣可以由使誤差函數最小化的思想中得到，一個很直觀的方法便是看misclassification的總數，但這樣會很容易讓我們的誤差函數變得不連續(太難搞啦！)，因此我們再考慮另一個誤差函數唄！有一個誤差函數叫`perceptron criterion(感知機準則)`，這個想法其實很好懂喔: 透過上面的公式，我們已經知道model輸出為1 or -1，我們現在用變量t來取代為一個`{1, -1}`的集合。對於c1類，我們可以得到輸出值大於0，而對於c2類我們可以得到輸出值小於0。這是不是也代表著，只要分類對了：  

<a href="https://www.codecogs.com/eqnedit.php?latex=w^T\phi&space;(xn)tn&space;>&space;0" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w^T\phi&space;(xn)tn&space;>&space;0" title="w^T\phi (xn)tn > 0" /></a>  
所以如果是誤分類的，我們就要盡量讓誤差越小越好：  

<a href="https://www.codecogs.com/eqnedit.php?latex=Ep(w)&space;=&space;-\sum_{n->M}^{}w^T\phi&space;(xn)tn" target="_blank"><img src="https://latex.codecogs.com/gif.latex?Ep(w)&space;=&space;-\sum_{n->M}^{}w^T\phi&space;(xn)tn" title="Ep(w) = -\sum_{n->M}^{}w^T\phi (xn)tn" /></a>  
這便是誤分類的data的誤差總和，`M`即為誤分類的集合。這樣我們也順利的將誤差函數的結果變為線性連續變化的了。下面是林軒田教授在上課講解的簡易`w`更新的公式：  
![](../screenshot/PLA1.png)  
`wt + 1`為修正後的權重，`t`則是代表第幾輪  
更細節的話：  

<a href="https://www.codecogs.com/eqnedit.php?latex=w^{\gamma&plus;1}&space;=&space;w^{\gamma}&space;-&space;\eta&space;\bigtriangledown&space;E_{p}(w)&space;=&space;w^\gamma&space;&plus;&space;\eta&space;\phi&space;_{n}t_{n}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?w^{\gamma&plus;1}&space;=&space;w^{\gamma}&space;-&space;\eta&space;\bigtriangledown&space;E_{p}(w)&space;=&space;w^\gamma&space;&plus;&space;\eta&space;\phi&space;_{n}t_{n}" title="w^{\gamma+1} = w^{\gamma} - \eta \bigtriangledown E_{p}(w) = w^\gamma + \eta \phi _{n}t_{n}" /></a>  
其中`η`為learning rate，`τ`為一個整數(代表的是次數，跟上面簡化版的t一樣)，這是一個隨機梯度下降的算法。

> 好的，我們來整理一下PLA的整個過程吧

我們不斷計算`y(x)`的值。如果分類正確，那麼`w`不變; 如果分類錯誤，我們會把向量`φ(xn)`加到`w`向量上if屬於c1類，或從中減掉`φ(xn)`if屬於c2類。  
<img src = "../screenshot/pla_process.png" width = "50%">  
我已經為大家列上圖片的順序囉！第一張圖的黑色箭頭作為初始的`w`向量，相應的黑色直線則為decision boundary。綠色的點就是我們選的第一個要糾正的點，對於紅色c1類。我們要將他的特徵向量加到我們的`w`上，因次新的decision boundary變動成第二張圖。第三張圖我們再次選了一個誤分類的點，一樣作了變化，整個過程後就能得到正確的decision boundary囉！  

> PLA怎麼停下來？ convergence theorem  

每一次修正的`wt+1`都會離目標權重`wf`更近，數學觀念上就是內積越來越大  
我們可能會懷疑內積越來越大不是因為兩線的夾角變小，而是因為向量的長度變大  
但通過證明發現，我們的mistake反而限制了我們向量長度的生長，因此長度對內積的影響並不會大到哪裡去  
而且根據公式，`兩個向量的夾角大於等於次數的根號與一個常數的product`  
因此T是有上限的 **(only if linear separable)**!