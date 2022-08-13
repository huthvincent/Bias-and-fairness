# 参考文献

----

**[1]Nuanced Metrics for Measuring Unintended Bias with Real Data for Text Classification**

[1]是从Jigsaw Unintended Bias in Toxicity Classification比赛给出的评价方式的论文找到的，提出了三个无意偏差的评价指标，根据不同子集计算AUC。

文中提到使用了两个数据集，一个来自维基评论，另一个为人工标注的数据集。第一个给出了链接，后者未给出。前者不是用来直接使用的，在文中他们根据原始数据集生成了新的用于测试的数据集。后者虽没有给出，但根据数据量和标注可猜测到应该和Kaggle是一样的数据集。给出了github项目链接，大概是维护原因原文中的项目链接已失效。人工划分的数据集当然相对来说更好，可以捕捉到一些多样化形式表示的身份标签，但可遇不可求。

**[2]Measuring and Mitigating Unintended Bias in Text Classification**

一种用来测量和减轻文本分类中的无意偏差的方法，用的数据集来自维基评论，与【1】相同。注意到这两篇文章作者交集比较多，应该是同一个实验室的成果。github项目中包含该论文的代码。



# [项目解析]([GitHub - conversationai/unintended-ml-bias-analysis](https://github.com/conversationai/unintended-ml-bias-analysis/))

____

项目中不包括[Wikipedia]([Wikipedia Talk Labels: Toxicity (figshare.com)](https://figshare.com/articles/dataset/Wikipedia_Talk_Labels_Toxicity/4563973))数据文件，需自行下载到`unintended-ml-bias-analysis-main\archive\data`中，部分文件确实，要正确运行代码有些需要从其他脚本中运行生成相关数据集。

### 代码解析

----

代码文件较多，在这里记录一些重要的相关代码文件。

`unintended-ml-bias-analysis-main\archive\unintended_ml_bias\Prep_Wikipedia_Data.ipynb`

- 通过`toxicity_annotated_comments.tsv和toxicity_annotations.tsv`两个维基评论的原始数据集生成训练集测试集和发展集（原文是dev，意义不明）。这个部分可以正常运行，在`data\`下生成`wiki_dev.csv、wiki_train.csv和wiki_test.csv`
- 通过前面划分的数据集，进而生成消除偏差的以及消除偏差的随机数据集（这个用来作为变量控制），需要用到`toxicity_debiasing_data_random.tsv和toxicity_debiasing_data.tsv`文件，这两个文件原项目中没有给出，一开始以为需要自从创建，因此花了很多时间猜测它的表结构，后来发现这两个文件是其他脚本生成的，没有注释或说明提到这点，真是误人子弟啊。



`unintended-ml-bias-analysis-main\archive\unintended_ml_bias\Dataset_bias_analysis.ipynb`

- 通过维基评论的两个原始数据集，分析数据集的平衡性，并合成平衡的数据集

- 这个文件生成了上述两个缺失的文件，但是这个文件无法正常运行起来。缺少`wikipedia_article_snippets_random.json`且代码编译错误。

  ```python
  AttributeError: 'dict' object has no attribute 'itervalues'
  将itervalues改为item解决
  
  AttributeError: 'tuple' object has no attribute 'as_series'
  AttributeError: 'tuple' object has no attribute 'total_deficit'
  暂未解决
  ```

- 该文件运行得很失败，不过问不大，不影响我们使用他们的数据集。



`unintended-ml-bias-analysis-main\archive\unintended_ml_bias\model_card.ipynb`

- 我们要用到的最重要的文件之一，这个文件演示了怎么计算合成数据集的预测值的几个指标。

- 输入`unintended-ml-bias-analysis-main\archive\unintended_ml_bias\model_car\intersectional_madlibs_scored.csv和unintended-ml-bias-analysis-main\archive\unintended_ml_bias\new_madlibber\input_data\en\en-us\words.csv`，前者是数据集和分数，后者是子群身份标签语料库。输出`unintended-ml-bias-analysis-main\archive\unintended_ml_bias\model_car\intersectional_results.csv和unitary_results.csv`,为不同模型的计算得到指标的值。

- 注意到下面的代码需要修改，原项目应该是改了结构忘了代码。

  ```python
  # 原项目下面的路径有误，我已修改
  madlibs_words = pd.read_csv('new_madlibber/input_data/en/en-us/words.csv')
  ```

- 核心算法封装在`model_tool.py`中，这个文件中将前者作为库进行调用。



`unintended-ml-bias-analysis-main\archive\unintended_ml_bias\Evaluate_Model.ipynb`

- 也是计算指标的评价，输入文件`unintended-ml-bias-analysis-main\archive\data\wiki_toxicity_test.csv`又缺失,代码简单量少，调用的`model_tool.py`。



`unintended-ml-bias-analysis-main\archive\unintended_ml_bias\model_tool.py`

- 计算三个指标的核心算法文件，内容复杂。
- 注意到他们使用的数据集没有标注身份标签，而且重新创建身份序列再划分，因此主要关注创建身份子集的内容，后续可以写入数据集中方便使用，计算指标的方法之前的方法Kaggle中也已经找到了。



### 数据集解析

-----

[Wikipedia]([Wikipedia Talk Labels: Toxicity (figshare.com)](https://figshare.com/articles/dataset/Wikipedia_Talk_Labels_Toxicity/4563973))原始数据文件：`toxicity_annotated_comments.tsv,toxicity_annotations.tsv,toxicity_worker_demographics.tsv`

`unintended-ml-bias-analysis-main\archive\data\toxicity_annotated_comments.tsv`

- 数据量159686条

- | 特征     | rev_id                | comment  | year           | logged_in      | ns                             | sample                          | split                                  |
  | -------- | --------------------- | -------- | -------------- | -------------- | ------------------------------ | ------------------------------- | -------------------------------------- |
  | **类型** | 整型                  | String   | 整型           | 布尔           | String                         | Srting                          | String                                 |
  | **说明** | 评论的ID,用于连接表用 | 评论内容 | 评论时间，无用 | 是否登录，无用 | 发表评论者是作者还是用户，无用 | 内容为random或blocked，意义不明 | 内容为train、test或dev，用于数据集划分 |

`unintended-ml-bias-analysis-main\archive\data\toxicity_annotated.tsv`

- 数据量1048575条

- | 特征     | rev_id                                                       | worker_id                                    | toxicity                                                     | toxicity_score                     |
  | -------- | ------------------------------------------------------------ | -------------------------------------------- | ------------------------------------------------------------ | ---------------------------------- |
  | **类型** | 整型                                                         | 整型                                         | 整型                                                         | 整型                               |
  | **说明** | 评论的ID,用于连接表用。这张表中rev_id不唯一，但（rev_id,worker_id）唯一。 | 应该是评论检查者的ID，每条评论有多个检查者。 | 0或1，表示这个检查者对评论的毒性判断，每条评论的所有检查点的毒性的平均值为评论的毒性。 | 连续型，意义不明，代码里没有用到。 |

- 该表与前表连接并稍作处理得到评论及其毒性值。

`unintended-ml-bias-analysis-main\archive\data\toxicity_worker_demographics.tsv`

- 数据量3591条

- | 特征 | worker_id | gender | english_first_language | age_group | education |
  | ---- | --------- | ------ | ---------------------- | --------- | --------- |

- 在数据处理上用不到，但我们可观察知该表是前表中检查者的信息，包括年龄学历等情况。



`Prep_Wikipedia_Data.ipynb`对维基评论数据处理后得到`wiki_dev.csv，wiki_train.csv和wiki_test.csv`，三个数据集的结构完全相同

`unintended-ml-bias-analysis-main\archive\data\wiki_train.csv`

- 数据量95669条

- | 特征     | rev_id             | toxicity                         | comment    | year             | logged_in        | ns                               | sample                            | split                         | is_toxic                                                  |
  | -------- | ------------------ | -------------------------------- | ---------- | ---------------- | ---------------- | -------------------------------- | --------------------------------- | ----------------------------- | --------------------------------------------------------- |
  | **类型** | 整型               | 浮点型                           | String     | 整形             | 布尔型           | String                           | String                            | String                        | 布尔型                                                    |
  | **说明** | 评论的ID，值唯一。 | 毒性值，通过之前的表平均值得到。 | 评论内容。 | 评论时间，无用。 | 是否登录，无用。 | 发表评论者是作者还是用户，无用。 | 内容为random或blocked，意义不明。 | 内容为train，用于数据集划分。 | 通过0.5的阈值对toxicity进行转换得到，表示该评论是否有毒。 |

- 我们使用时相比于维基评论的原始数据集应该可以用这个数据集更方便些。



`bias_madlibs_77k.csv,bias_madlibs_89k.csv,toxicity_fuzzed_testset.csv`都是专门用来测试评价模型偏差的测试数据集，前两个根据句子模板生成，两个数据集数据量不同，前者约77000条，后者约89000条。最后一个是根据维基评论生成的模糊数据集，评论更复杂具真实性。

`unintended-ml-bias-analysis-main\archive\unintended_ml_bias\eval_datasets\bias_madlibs_77k.csv`

- 数据量76564条

- | **特征** | Text       | Label                                    | Template                   |
  | -------- | ---------- | ---------------------------------------- | -------------------------- |
  | **类型** | String     | String                                   | String                     |
  | **说明** | 评论内容。 | 评论标签，Bad或Not_Bad，表示评论是否有毒 | 生成该评论的句子模板，无用 |

`unintended-ml-bias-analysis-main\archive\unintended_ml_bias\eval_datasets\bias_madlibs_89k.csv`

- 数据量89483

- | 特征     | Text       | Label                                    |
  | -------- | ---------- | ---------------------------------------- |
  | **类型** | String     | String                                   |
  | **说明** | 评论内容。 | 评论标签，Bad或Not_Bad，表示评论是否有毒 |

`unintended-ml-bias-analysis-main\archive\unintended_ml_bias\eval_datasets\toxicity_fuzzed_testset.csv`

- 数据量1492条

- | 特征     | rev_id             | comment    |              toxic |
  | -------- | ------------------ | ---------- | -----------------: |
  | **类型** | 整型               | String     |             布尔型 |
  | **说明** | 评论的ID，值唯一。 | 评论内容。 | 表示评论是否有毒。 |



`intersectional_madlibs_scored.csv和intersectional_results.csv`是`model_card.ipynb`演示中的输入文件和输出结果

`unintended-ml-bias-analysis-main\archive\unintended_ml_bias\model_card\intersectional_madlibs_scored.csv`

- 数据量30240条

- | 特征     | template         | toxicity                                | phrase   | TOXICITY@n                                |
  | -------- | ---------------- | --------------------------------------- | -------- | ----------------------------------------- |
  | **类型** | String           | String                                  | String   | 浮点型                                    |
  | **说明** | 生成该句子的模板 | 内容为nontoxic或toxic，表示句子是否有毒 | 句子内容 | 这里有6列，n=1~6，表示6种方法的毒性预测值 |

`unintended-ml-bias-analysis-main\archive\unintended_ml_bias\model_card\intersectional_results.csv`

- 数据量255条

- | 特征     | subgroup           | TOXICITY@n_subgroup_auc               | TOXICITY@n_bpsn_auc | TOXICITY@n_bnsp_auc |
  | -------- | ------------------ | ------------------------------------- | ------------------- | ------------------- |
  | **类型** | String             | 浮点型                                | 浮点型              | 浮点型              |
  | **说明** | 划分子群的身份标签 | n=1~6,表示6种方法的各个指标计算结果。 |                     |                     |

  
