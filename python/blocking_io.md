# 如何在协程中嵌入阻塞函数
 
## 应用场景
 
在基于协程的python程序中，有时候需要重新开启一个阻塞线程来执行监听全局的状态，例如Watchdog，flask-server等，甚至直接在自己的程序当中添加While True，来做一些非常傻但有用的代码，此时就涉及到如何在不影响主线程执行的情况下另开一个线程来执行该阻塞程序。  

阻塞在代码层面分为同步阻塞和异步阻塞，表现出来的就是: 

- 同步阻塞 
```python
def sync_func(): 
    pass
```

- 异步阻塞 
```python
def async_func(): 
    pass
```

## 问题定义 

阻塞程序主要有以下四个分类： 

1. 在同步程序中执行同步阻塞 
1. 在同步程序中执行异步阻塞 
1. 在异步程序中执行同步阻塞 
1. 在异步程序中执行异步阻塞 
 
关于问题1，可通过传统的thread和multiprocess来解决；关于问题2，我不想说，因为这种场景很少见，也可转化成其他问题来解决。所以在本篇文章中，主要以针对问题3、4来讲解。 

所有的问题描述完全没有直接上代码来的直接，请直接阅读以下代码，并尝试将其跑通。 

```python
async def bot_start(): 
    await http_service() 
    while True: 
        await asyncio.sleep(1) 
        print('bot tick...') 
 
async def http_service(): 
    # start flask or aiohttp web framework 
    while True: 
    await asyncio.sleep(1) 
    print('http service tick ...') 
 
async def main(): 
    await bot_start() 
 
asyncio.run(main()) 
``` 

> 为了简化问题，我把所有的阻塞全部设置成while True。 

很明显，执行结果一直是http service tick … 

这里需要说明以下，flask是不支持异步的web框架，而aiohttp是建立在协程之上的，所以两者非常具有代表性，且解决方案不同。 

## 同步阻塞 

同步阻塞函数的问题可以描述成以下代码： 

```python
import asyncio 
import time 
import threading 
from datetime import datetime 
from concurrent import futures 
 
def bot_start(): 
    while True: 
        time.sleep(1) 
        print('bot sleep on ...') 
 
def http_service(): 
    while True: 
        time.sleep(1) 
        print('http service tick ...') 
 
async def main(): 
    bot_start() 
    http_service() 
 
asyncio.run(main()) 
```
 
由于两个阻塞函数是同步的，故只能沿用传统多线程/进程的方式来执行，才可以实现并行执行。方法如下： 

```python
import asyncio 
import time 
import threading 
from datetime import datetime 
from concurrent import futures 
 
 
def bot_start(): 
    while True: 
        time.sleep(1) 
        print('bot sleep on ...') 
 
async def event_bus(): 
    loop = asyncio.get_event_loop() 
    executor = futures.ThreadPoolExecutor(max_workers=2) 
    task1 = loop.run_in_executor(executor, bot_start) 
    task2 = loop.run_in_executor(executor, http_service) 
    await asyncio.wait([task2, task1]) 
    print('task done') 
  
def http_service(): 
    while True: 
        time.sleep(1) 
        print('http service tick ...') 
 
async def main(): 
    await event_bus() 
 
asyncio.run(main())
```
 
这里是使用EventLoop中的run_in_executor的方式来执行，从而可以将不同的阻塞函数放到不同的线程当中去执行。这里使用的ThreadPoolExecutor，大家也可以试一试ProcessPoolExecutor，使用方法一模一样，详细用法请看asyncio.loop.run_in_executor 

至此，已经可以在协程当中执行两个同步阻塞函数了，http_service函数的代码替换成flask app.run() 代码也是可以执行的，毕竟原理一致。 

## 异步阻塞 

如果在协程当中添加一个异步阻塞的话，代码更简单，毕竟协程本身就是异步编程。话不多说，先来看大家如何来改造以下代码，让其能够正常执行。 

```python
import asyncio 
import time 
import threading 
from concurrent import futures 
 
 
async def bot_start(): 
    while True: 
        await asyncio.sleep(1) 
        print('bot tick ...') 
 
async def http_service(): 
    while True: 
        await asyncio.sleep(1) 
        print('http service tick ...') 
 
async def main(): 
    await http_service() 
    await bot_start()  
 
asyncio.run(main()) 
```

这里很明显，当在bot_start函数之前执行http_service函数，此时的EventLoop中还没调度完其任务，且处于死循环当中，只能够无限等待。 

那么该问题如何解决呢？ 

其实核心的思想还是与上类似，需要在多个线程/进程当中来执行。通过查阅asyncio文档可知，可通过run_coroutine_threadsafe函数可创建新线程并执行异步阻塞的代码。故解决方案如下： 

```python
import asyncio 
import time 
import threading 
from concurrent import futures 
 
 
async def bot_start(): 
    while True: 
        await asyncio.sleep(1) 
        print('bot tick ...') 
 
async def http_service(): 
    while True: 
        await asyncio.sleep(1) 
        print('http service tick ...') 
 
async def main(): 
    loop = asyncio.get_event_loop() 
    asyncio.run_coroutine_threadsafe(http_service(), loop=loop) 
    await http_service() 
    await bot_start()  
 
asyncio.run(main()) 
```
 

以上最核心的代码就是： 
```python
loop = asyncio.get_event_loop() 
asyncio.run_coroutine_threadsafe(http_service(), loop=loop)
```

首先获取当前正在的event_loop对象，然后使用run_coroutine_threadsafe函数来构造基于协程的线程安全方法，即可在此将http_service协程对象放到新的线程上面去执行。  

还是原来的配方，还是原来的方法。 

## 总结 

1. 同步阻塞使用thread/process executor 
1. 异步阻塞使用run_coroutine_threadsafe 

原理还是使用不同线程来完成并行执行的功能。 

参考文章： 

- https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.run_in_executor 
- https://carlosmaniero.github.io/asyncio-handle-blocking-functions.html 
- https://pymotw.com/3/asyncio/executors.html 

 