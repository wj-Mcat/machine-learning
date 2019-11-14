# numpy 笔记

- 初始化一个0矩阵

  ```python
  matrix = np.zeros((10,2))
  ```

- 更改数据类型

  ```python
  data = np.random.rand(2,3,dtype=np.float)
  print(data.dtype)
  # 不能直接需改 data.dtype属性，而是需要使用 `astype`函数
  data.astype(np.double)
  ```

- 

