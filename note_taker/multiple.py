import asyncio

async def worker(i):
    await asyncio.sleep(1)
    print(f"done {i}")
    return i

async def main():
    tasks = [asyncio.create_task(worker(41)), asyncio.create_task(worker(64))]

    print("other work continues here...")

    results = await asyncio.gather(*tasks)  # acknowledgement point
    # print("all done:", results)
    a,b = results
    print(a,b)

asyncio.run(main())
