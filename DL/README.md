## 什麼是Neural Network?
不同的Neuron合併或串接在一起組成Neural network  
每一個neuron都是由logistic regression組成  
不同的串接方式會有不同的網路架構  
回去看看我們的logistic regression有哪些參數：`weight`，`bias`  
因此neural network的參數便是所有neuron的`weight`和`bias`  

在傳統訓練model中，feature engineering很重要  
但對於NN而言，即使把所有feature直接丟進去也能訓練出個結果  
但NN要設計網路架構卻非常吃經驗  
所以現在挑戰變成 **feature engineering v.s. NN structure design**  
所以不是說DL真的比較香  
而是看你覺得哪個問題比較容易解決？  
e.g. image recognition v.s. NLP 的話會怎麼抉擇呢？  
image recognition在NN上效果非常好  
因為人類無法理解pixel之類的feature  
因此feature engineering也很不容易  
而NLP的feature人類的理解程度反而更好  
因此手動作feature engineering可以得到不錯的結果

## CNN
### 閱讀清單
* [ML Lecture 10: Convolutional Neural Network by 李宏毅](https://www.youtube.com/watch?v=FrKWiRv254g&t=1s&ab_channel=Hung-yiLee)
* [A Beginner's Guide To Understanding Convolutional Neural Networks by Adit Deshpande](https://adeshpande3.github.io/adeshpande3.github.io/A-Beginner's-Guide-To-Understanding-Convolutional-Neural-Networks/)

### 小知識點
* 為什麼需要CNN？  
  在全連結的類神經網路中，我們需要更新跟輸入平方size甚至更多的權重，時間複雜度非常得高。然而我們真的需要每次都更新全部的權重嗎？
* CNN的流程  
  input -> (convolution layer -> max pooling) * n -> flatten
* convolution layer  
  在卷積層裡，由於我們要檢查的pattern通常只佔整張image的小部分。假設一張5x5的圖片裡，我們要檢測的pattern只有3x3。那我們就會用3x3的filter，從左上要右下一步步進行矩陣相乘，每一步的大小由stride決定，而3x3代表一次會更新9個權重。因此同樣的pattern，不管出現在圖片的哪個地方，都能用同一個filter檢測。But, 若是不同大小的pattern，將不能用同一個filter檢測！
* input/filter/output可以有好幾個channel  
  這些矩陣不單可以只是二維的，也可以是三維的！舉個🌰，彩色圖片我們將不同把圖片用二進制顯示，將由R channel的matrix，G channel的matrix，和B channel的matrix疊加而成。那我們也需要三個filter各別作用在三個channel，輸出也會有三個channel。 (**filter有多少channel，output就會有多少channel**)
* 給自己挖坑想要自己手刻  
  train的過程很難寫，但基本核心思想就是部分的weight將永遠是0
* Max Pooling  
  卷積層輸出之後，作subsampling讓輸出的size變少。取值的方式可以是範圍的matrix中取maximum value或是average value。
* Flatten  
  最後一步，把feature map拉直，送進全連結的類神經網路，作跟以前一模一樣的事。這樣時間複雜度已經比從頭到尾都用全連結的NN作更新還要減少很多！

## BP反向傳播更新weight
[back propagation](../back-propagation/README.md)