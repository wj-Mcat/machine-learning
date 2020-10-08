# pandas

- dropna

  顾名思义

- 将数据进行分类

  ```python
  train_set = data.sample(frac = 0.5,random_state = 100)
  test_set  = data.drop(train_set.index)
  ```

- describe & transpose

  ```python
  train_stats = train_dataset.describe()
  print(train_stats)
  train_stats = train_stats.transpose()
  print(train_stats)
  ```

- 随意去几个样本

  ```python
  data.sample(10)
  ```

  

