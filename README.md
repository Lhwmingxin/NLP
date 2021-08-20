## 一、 BERT
### 1. 什么是BERT
#### 1）BERT简介：
 &emsp;  &emsp; 本文主要介绍一个名为BERT的模型。BERT与现有语言模型不同的是，BERT旨在通过调节所有层中的上下文来进行深度双向的预训练。因此，预训练的BERT表示可以通过另外的输出层进行调整，以创建用于广泛任务的状态模型，例如问题转换和语言参考，而无需实质的任务特定体系结构修改。 
 &emsp;  &emsp; BERT全称是“Bidirectional Encoder Representation from Transformers“，即双向Transformer解码器。是一种NLP领域的龙骨模型，用来提取各类任务的基础特征，即作为预训练模型为下游NLP任务提供帮助。
#### 2）BERT的创新点
  &emsp; &emsp;  <font color=RED>BERT并没有过多的结构上的创新点，BERT依然采用的是transformer的结构。BERT模型有四大关键词:预训练，深度，双向转换，语言理解</font>

-  	**预训练：** BERT是通过在无标注的数据上进行训练的预训练模型，所以泛化能力较强。BERT的预训练过程包含两个任务一个是Masked Language Model，还有一个是Next Sentence Prediction。
- **深度:** 最开始官网给了两个版本Base和Large两个版本的BERT模型，供所有人使用
 &emsp; Base:版本Layer = 12, Hidden = 768, Head = 12, Total Parameters = 110M
 &emsp; Large版本:Layer = 24, Hidden = 1024, Head = 16, Total Parameters = 340M
 &emsp; 对比于原始论文的Transformer: Layer = 6, Hidden = 2048, Head = 8，可以看出Bert是一个深而窄的模型，效果更好。但是由于transformer的残差（residual）模块，层数并不会引起梯度消失等问题，但是并不代表层数越多效果越好，有论点认为低层偏向于语法特征学习，高层偏向于语义特征学习。
- **双向转换：** BERT的模型架构基于多层双向转换解码。Bert直接引用了Transformer架构中的Encoder模块，并舍弃了Decoder模块, 这样便自动拥有了双向编码能力和强大的特征提取能力。“双向”表示模型在处理某一个词时，它能同时利用前面的词和后面的词两部分信息。“双向”的训练方式为随机遮蔽输入词块的某些部分，然后仅预测那些被遮蔽词块。我们将这个过程称为“遮蔽LM”(MLM)
- **语言理解：**  更加侧重语言的理解，而不仅仅是生成(Language Generation)

### 2. BERT的预训练过程
#### 1）Masked Language Model ( MLM，带mask的单词级别语言模型训练 )
 &emsp;  &emsp; MLM类似完形填空，文章作者在一句话中随机选择 15% 的词汇用于预测。对随机Mask处理的单词，用非监督学习的方法去预测mask位置的词。对于在原句中被抹去的词汇， 80% 情况下采用一个特殊符号 [MASK] 替换， 10% 情况下采用一个任意词替换，剩余 10% 情况下保持原词汇不变。

##### 实际操作
在BERT中, Masked LM(Masked language Model)构建了语言模型, 这也是BERT的预训练中任务之一, 简单来说, 就是随机遮盖或替换一句话里面任意字或词, 然后让模型通过上下文的理解预测那一个被遮盖或替换的部分, 之后做$Loss$的时候只计算被遮盖部分的$Loss$, 其实是一个很容易理解的任务, 实际操作方式如下:

 

 1. 随机把一句话中$15 \%$的$token$替换成以下内容:
        1) 这些$token$有$80 \% $的几率被替换成$[mask]$;
        2) 有$10 \%$的几率被替换成任意一个其他的$token$;
        3) 有$10 \%$的几率原封不动.
  2. 之后让模型预测和还原被遮盖掉或替换掉的部分, 模型最终输出的隐藏层的计算结果的维度是:
        $X_{hidden}: [batch\_size, \ seq\_len, \ embedding\_dim]$
        我们初始化一个映射层的权重$W_{vocab}$:
        $W_{vocab}: [embedding\_dim, \ vocab\_size]$
        我们用$W_{vocab}$完成隐藏维度到字向量数量的映射, 只要求$X_{hidden}$和$W_{vocab}$的矩阵乘(点积):
        $X_{hidden}W_{vocab}: [batch\_size, \ seq\_len, \ vocab\_size] $ 之后把上面的计算结果在$vocab\_size$(最后一个)维度做$softmax$归一化,
    是每个字对应的$vocab\_size$的和为$1$, 我们就可以通过$vocab\_size$里概率最大的字来得到模型的预测结果,
    就可以和我们准备好的$Label$做损失($Loss$)并反传梯度了.
        注意做损失的时候, 只计算在第1步里当句中随机遮盖或替换的部分, 其余部分不做损失, 对于其他部分, 模型输出什么东西, 我们不在意.
        
##### 相关问题及解答：
Q1：为什么要使用mask？
A1：BERT只使用了transformer模型的 Encoder 结构。而 Encoder 的 Self Attention 层，每个 token 会把大部分注意力集中到自己身上，那么这样将容易预测到每个 token，模型学不到有用的信息。所以BERT 提出使用 mask，把需要预测的词屏蔽掉。

--------
Q2：为啥要以一定的概率使用随机词呢？（原文例子为：my dog is hairy → my dog is [MASK]）
A2：这是因为transformer要保持对每个输入token分布式的表征，否则Transformer很可能会记住这个[MASK]就是"hairy"。至于使用随机词带来的负面影响，文章中解释说,所有其他的token(即非"hairy"的token)共享15%*10% = 1.5%的概率，其影响是可以忽略不计的。Transformer全局的可视，又增加了信息的获取，但是不让模型获取全量信息。

----------------
Q3：为什么不把15%的词全部 [MASK] 替换？
A3：在在下游的自然语言处理任务中，语句中并不会出现 [MASK] 标记，[MASK]仅仅只是为了训练。因此，为了和后续任务保持一致，应该按比例在需要预测的词的位置输入原词、随机词或者[MASK]。
 &emsp;  &emsp; 而且这么做的另一个好处是：预测一个词汇时，模型并不知道输入对应位置的词汇是否为正确的词汇（ 10% 概率），哪些单词被遮掩成了[MASK]，哪些单词被替换成了其他单词。正是在这一种高度不确定的情况下, 反倒逼着模型快速学习该token的分布式上下文的语义, 迫使模型更多地依赖于上下文信息去预测词汇，尽最大努力学习原始语言说话的样子!!! 同时因为原始文本中只有15%的token参与了MASK操作, 所以并不会破坏原语言的表达能力和语言规则!!!并且赋予了模型一定的纠错能力（ BERT 模型 [Mask] 标记可以看做是引入了噪音）
 &emsp;  &emsp; 其实这样做还有另外一个缺点，就是每批次数据中只有 15% 的标记被预测，这意味着模型可能需要更多的预训练步骤来收敛。
#### 2) Next Sentence Prediction(句子级别的连续性预测任务)
 &emsp; &emsp; 许多重要的下游任务，如问答（QA）和自然语言推理（NLI）都是基于理解两个句子之间的关系，这并没有通过语言建模直接获得。 
 &emsp;  &emsp;  BERT考虑到问答系统，智能聊天机器人之类的任务场景，所以增加了第二个任务，即预测输入 BERT 的两段文本是否为连续的文本。引入这个任务可以更好地让模型学到连续的文本片段之间的关系。
 &emsp; &emsp; 在为了训练一个理解句子的模型关系，预先训练一个二进制化的下一句测任务，这一任务可以从任何单语语料库中生成。具体地说，当选择句子A和B作为预训练样本时，B有50％的可能是A的下一个句子，也有50％的可能是来自语料库的随机句子。
 &emsp; &emsp; 团队完全随机地选择了NotNext语句，最终的预训练模型在此任务上实现了97％-98％的准确率。[4]


##### 实际操作

   

 1. 首先我们拿到属于上下文的一对句子, 也就是两个句子, 之后我们要在这两段连续的句子里面加一些特殊$token$:
        $[cls]$上一句话,$[sep]$下一句话.$[sep]$
        也就是在句子开头加一个$[cls]$, 在两句话之中和句末加$[sep]$, 具体地就像下图一样:
    
  2. 我们看到上图中两句话是$[cls]$ my dog is cute $[sep]$ he likes playing $[sep]$, $[cls]$我的狗很可爱$[sep]$他喜欢玩耍$[sep]$, 除此之外, 我们还要准备同样格式的两句话,
    但他们不属于上下文关系的情况;
        $[cls]$我的狗很可爱$[sep]$企鹅不擅长飞行$[sep]$, 可见这属于上下句不属于上下文关系的情况;
        在实际的训练中, 我们让上面两种情况出现的比例为$1:1$, 也就是一半的时间输出的文本属于上下文关系, 一半时间不是.
        我们进行完上述步骤之后, 还要随机初始化一个可训练的$segment \ embeddings$, 见上图中, 作用就是用$embeddings$的信息让模型分开上下句, 我们一把给上句全$0$的$token$, 下句啊全$1$的$token$,
    让模型得以判断上下句的起止位置, 例如:
        $[cls]$我的狗很可爱$[sep]$企鹅不擅长飞行$[sep]$
        $0 \quad \ 0 \ \ 0 \ \ 0 \ \ 0 \ \ 0 \ \ 0 \ \ 0 \ \ \ 1 \ \ 1 \ \ 1 \ \ 1 \ \ 1 \ \ 1 \ \ 1 \ \ 1$
        上面$0$和$1$就是$segment \ embeddings$.
  3. 还记得我们上节课说过的, 注意力机制就是, 让每句话中的每一个字对应的那一条向量里, 都融入这句话所有字的信息, 那么我们在最终隐藏层的计算结果里, 只要取出$[cls]token$所对应的一条向量, 里面就含有整个句子的信息,
    因为我们期望这个句子里面所有信息都会往$[cls]token$所对应的一条向量里汇总:
        模型最终输出的隐藏层的计算结果的维度是:
        我们$X_{hidden}: [batch\_size, \ seq\_len, \ embedding\_dim]$
        我们要取出$[cls]token$所对应的一条向量, $[cls]$对应着$\ seq\_len$维度的第$0$条:
        $cls\_vector = X_{hidden}[:, \ 0, \ :]$
        $cls\_vector \in \mathbb{R}^{batch\_size, \ embedding\_dim}$
        之后我们再初始化一个权重, 完成从$embedding\_dim$维度到$1$的映射, 也就是逻辑回归, 之后用$sigmoid$函数激活, 就得到了而分类问题的推断.
        我们用$\hat{y}$来表示模型的输出的推断, 他的值介于$(0, \ 1)$之间:
        $\hat{y} = sigmoid(Linear(cls\_vector)) \quad \hat{y} \in (0, \ 1)$

### 3.图解BERT

![在这里插入图片描述](https://img-blog.csdnimg.cn/b27ffc7ede3c4e8d97397a3f951771e5.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTg1NjE3MA==,size_16,color_FFFFFF,t_70)


 &emsp;  &emsp;  具体的，如上图。在进行mask LM时，sentenceA起始位置字符为[CLS]，它 的含义是分类（class的缩写）。[CLS]对应的输出向量T是整个句子的embedding，可以作为文本分类时，后续分类器的输入。特殊符[SEP]是用于分割两个句子。在最后一个句子的尾部也会加上[SEP] token。训练中，sentenceA的中的每个词对应的句子向量都是sentenceA。

 &emsp;  &emsp;  就像 Transformer 中普通的 Encoder 一样，BERT 将一串单词作为输入，这些单词在 Encoder 的栈中不断向上流动。每一层都会经过 Self Attention 层，并通过一个前馈神经网络，然后将结果传给下一个 Encoder。![在这里插入图片描述](https://img-blog.csdnimg.cn/cca6322f7a9f49d790e5c56bdf6d6e6d.webp?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTg1NjE3MA==,size_16,color_FFFFFF,t_70#pic_center)


 &emsp;  &emsp;  以下图句子分类为例，我们将BERT的预训练结果，输入一个分类器（上图中的 Classifier，属于监督学习）进行训练。在训练过程中 几乎不用改动BERT模型，只根据任务训练分类器就行，这个训练过程即为微调。
![BERT句子分类](https://img-blog.csdnimg.cn/d2994902f1dd4f6fbd8dcd5ef362b436.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTg1NjE3MA==,size_16,color_FFFFFF,t_70)
### 4.模型输入与输出
&emsp;  &emsp;  BERT 模型的主要输入是文本中各个字/词(或者称为 token)的原始词向量，该向量既可以随机初始化，也可以利用 Word2Vector 等算法进行预训练以作为初始值；输出是文本中各个字/词融合了全文语义信息后的向量表示。
#### 1) 模型输入
 &emsp;  &emsp;  BERT的输入可以是单一的一个句子或者是句子对，实际的输入值是segment embedding与position embedding相加。

 &emsp;  &emsp;  BERT的输入词向量是三个向量之和：

-	词嵌入向量（word Embedding）：WordPiece tokenization subword词向量。
-	语句向量（Segment Embedding）：表明这个词属于哪个句子（NSP需要两个句子）。
-	位置编码向量（Position Embedding）：学习出来的embedding向量。这与Transformer不同，Transformer中是预先设定好的值。

![在这里插入图片描述](https://img-blog.csdnimg.cn/2449a6bbfda9492290c601e2b4b7eae3.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTg1NjE3MA==,size_16,color_FFFFFF,t_70)
#### 2) 模型输出
每个位置输出一个大小为 hidden_size（在 BERT Base 中是 768）的向量。对于上面提到的句子分类的例子，我们只关注第一个位置的输出（输入是 [CLS] 的那个位置），第一个位置输出的向量作为后面分类器的输入。
### 5. 词嵌入（Embedding）
#### 1) 词嵌入概念
单词不能直接输入机器学习模型，而需要某种数值表示形式，以便模型能够在计算中使用。在上一章节的学习中我们详细讲述了 这一方面内容。经过实验证明，相比于在小规模数据集上和模型一起训练词嵌入，更好的一种做法是，在大规模文本数据上预训练好词嵌入，然后拿来使用。

#### 2）elom
elmo：将上下文当作特征，没有对每个单词使用固定的词嵌入，而是在为每个词分配词嵌入之前，查看整个句子，融合上下文信息，但是无监督的语料和我们真实的语料还是有区别的，不一定的符合我们特定的任务，是一种双向的特征提取。
![在这里插入图片描述](https://img-blog.csdnimg.cn/cb1d0dea04844cd6b73a809b091580cc.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L3dlaXhpbl80NTg1NjE3MA==,size_16,color_FFFFFF,t_70)
## 二、GPT



### 参考文献
[1] [BERT入门（有空再补）](https://blog.csdn.net/qq_56591814/article/details/119781554)
[2] [BERT：用于语义理解的深度双向预训练转换器(Transformer)](https://carrylaw.github.io/anlp/2018/11/07/nlp14/)
[3] [一文读懂BERT(原理篇)](https://blog.csdn.net/jiaowoshouzi/article/details/89073944)
[4]  [论文原文:BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
