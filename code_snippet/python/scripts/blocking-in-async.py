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
    # await http_service()
    await bot_start()


asyncio.run(main())
