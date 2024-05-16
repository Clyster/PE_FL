# PE_FL
> unzip part.zip before running run.sh

jieguo jieguo2 baseline的表现

jieguo3 错误改动

jieguo4 5 20%模态丢失
### 0507 
resume的ckpt会在所有循环外载入到global weights中，相当于初始化的值。
* 进入global循环后初始化local权重，抽取幸运用户；
  *   进入每个user训练循环后，先初始化模型和优化器，载入global权重（如果权重长度为0则不载入），载入成功后再data parallel（目前没有多gpu用不上其实），然后根据user传参数给data loader来读取对应的数据。在这之后进入每个epoch训练，每次迭代之后会我去除了原本判断是否是最佳结果的判断句式。
  *   训练完5轮之后，保存loacal权重
* 训练完当前批次幸运用户之后，计算新的global weights。然后根据这个global weights来进行之前被删掉的best result的判断。保存结果。进入下一个global循环

### 0510
main3.py kittiloader2.py模拟了rgb和depth单一丢失的情况

### 515
从现在开始更改日志的方式，会更加详细的说明改动

在这之前的所有工作：

* learn rate都是1e-3
* 先是测试的100%数据集的情况下，模拟丢失20%和33%的rgb或者depth输入的情况下的log和不丢失的log
* 然后是下采样（50%）数据集，模拟丢失20%的rgb或者depth输入的情况下的log

之后要调整更多参数！

### 516
在下采样的情况下测试benchmark
