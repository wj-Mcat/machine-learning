# 记录常见的小技巧

- 展示图片的灰度图片

  ```python
  plt.imshow(show_data[index],cmap=plt.cm.gray_r)
  ```

- 设置`5*5`的图片格局

  ```python
  plt.figure(figsize=(20,20))  # 设置每张图片的像素大小
  
  plt.subplot(5,5,index+1)	 # 设置5*5 方格中每张图片的索引
  ```

- pairplot

  ```python
  import seaborn as sns
  sns.pairplot(train_set[["MPG", "Cylinders", "Displacement", "Weight"]])
  plt.show()
  ```

- 画树状图

  ```python
  plt.hist(data['Log1pSalary'],bins=100)
  ```

  - bins

    表示图形中树状图的个数多少，数字越大，每个树枝就越细

  - 这个和条形图是有区别的，条形图是散开的，这个是紧密挨在一起的