import asyncio
import time
import threading
from datetime import datetime
from concurrent import futures


def bot_start():
    while True:
        time.sleep(1)
        print('bot tick ...')

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
