## 維度災難
先複習一下overfit的解決辦法：  
1. sample數++
2. 正則化，限制參數空間
3. 降維
4. cross validation

真實世界中，為了完美得分類，我們需要取很多樣的特徵。然而，分類器的效能在過了最適當的特徵數後，是會隨特徵數(維 度)下降的！  

為什麼呢？這裡用兩個角度解釋：

同樣的樣本數，投射到更高維度的空間中後，分佈會變得更稀疏(樣本密度呈現指數級下降)。也因為變得更稀疏，所以我們容易找到一個hyplane作完美分類。但也因為我們在高維度能得到一個線性分類的結果，相當於在低維度使用了非線性分類，我們的classifier連同noise以及outlier也學習了，結果就是我們的泛化能力變很差！

> 泛化(generalization)：對於樣本空間外的數據預測

儘管我們在training set上看到的結果：非線性的分類效果比較好，但是線性的泛化效果好很多。這裡用另一個角度來做解釋：剛剛提過投射到高維度時，同樣的樣本數量密度會指數級下降，如果要覆蓋同樣的特徵空間，那所需的樣本數也會變多！而且，距離的度量在越高維度也會顯得越沒有意義，然而有很多classifier都是基於距離來做分類的。  

為了避免維度災難：  
除了樣本數要呈現指數級上升，對於non-linear classifier，維度就不能太高 e.g. neural network, knn, decision tree。對於linear classifier，可以使用較多特徵 e.g. bayesian classifier。  

那要選哪些特徵呢？  
Feature Selection: greedy, best-first, etc  
Feature Extraction: PCA

## PCA (Principal Component Analysis)
關於PCA的數學與基本原理，二話不說推這篇：  
* [PCA的数学原理](http://blog.codinglabs.org/articles/pca-tutorial.html)

這篇非常詳細的介紹pca的"發明過程"：  
* 從inner product(內積) 介紹 basis(基) 的應用
* 平常我們用的2D座標，也只是向量在兩個基上的投影
* 換句話說，要投影到其他基，甚至投影到幾個基也都是可行的！
* 如何選擇基呢？
* 我們會想讓他們投影到基上，不會重疊(也就是information loss)
* 數學上也就是在同一個基上的variance(方差)越大
* 就多個基相互之間的關係，我們會希望covariance(協方差)為0 (訊息不重疊)
* 發現奇蹟：我們可以很容易得從原始矩陣得到協方差矩陣
* 根據上面variance和covariance的要求，我們會希望協方差矩陣是個對角線矩陣
* 我們要求一個p矩陣，它可以讓協方差矩陣對角化
* p: 協方差矩陣的特徵向量單位化的列向量
* 這個p，其實就是我們要找的基啦！
* 用p去跟原始矩陣作運算就可以得到投影後的位置囉！

PCA的一些缺點：  
* PCA假設feature主要分佈在正交的方向上，但如果非正交的方向上也存在幾個variance較大的，那PCA就無法應付了
* PCA用來消除linear dependance，但對於高階的linear dependance，只能用kernal PCA了
* PCA無參數，也就是沒法客製化的調參