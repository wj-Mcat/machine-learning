# sklearn 中好用的辅助工具

- train_test_split

  用于对训练样本进行测试集合训练集进行分割

  ```python
  from sklearn.model_selection import train_test_split
  
  x_train,x_validation ,y_train,y_validation = train_test_split(x_sample,y_sample,train_size=0.*,random_state = 100)
  ```

- GridSearchCV

  这个方法是真的好用，能否配合keras来使用，那就更爽了

  

