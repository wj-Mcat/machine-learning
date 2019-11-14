# 奇淫技巧

## 获取命令行执行结果

```python
import subprocess
test = subprocess.Popen(["python","test.py"], stdout=subprocess.PIPE)
output = test.communicate()[0]
print(output)
```

