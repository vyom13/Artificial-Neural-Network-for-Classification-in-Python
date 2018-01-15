# Artificial Neural Network for Classification in Python
**Artificial Neural Network (ANN) for classification**  
 Modify  the  python  script  you  have  written  for  doing  regression  using  ANN  in homework  assignment  2  so  that  your  modified  code  can  perform  classificationanalysis.The major difference between using ANN for regression and using ANN for clas-sification  lies  in  the  cost  function.   For  classification,  use  the  cross-entropy  costfunction as follows:  
 
 *J(Θ) =−1m[m∑i=1K∑k=1y(i)klog(hΘ(x(i)))k+(1−y(i)k)log(1−(hΘ(x(i)))k)]*  
 
 where y is the actual output and hθ(x) is the predicted output.Your modified ANN will contain one input layer, one hidden layer, and one output layer.   The  input  layer  has  two  units,  the  hidden  layer  has  two  units,  and  theoutput layer has one unit.  For this particular ANN architecture,K= 1. With the cross-entropy cost function, your calculation of  
 
 *∂∂a(L)J(Θ)*  
 
 should be modified accordingly. L is the total number of layers in the ANN. Another modification you need to make to the ANN python script you have written for homework 2 is to sum the partial derivative  
 
 *∂∂θ(l)ijJ(Θ)*  
 
 over all training samples. Refer to slide #43 in the neural network lecture slides. Apply the resulting ANN python script to perform a binary classification of the virginica  and  versicolor  flowers  in  the  iris  dataset  using  petal  length  and  petal width.  Specifically, you will perform a leave-one-out analysis by using one flowerfor testing and the remaining 99 flowers for training. If the testing result is differentfrom the actual flower type, the error is 1.0.  Otherwise, the error is 0.  Performthis leave-one-out analysis 100 times and get the average error rate.
