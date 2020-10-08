import asyncio
import time
import threading
from concurrent import futures


async def bot_start():
    await http_service()
    while True:
        await asyncio.sleep(1)
        print('bot tick ...')

async def http_service():
    # start flask or aiohttp web framework
    while True:
        await asyncio.sleep(1)
        print('http service tick ...')


async def main():
    await bot_start()

asyncio.run(main())
