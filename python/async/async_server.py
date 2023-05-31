
#
## This is directly copied from:
##   https://www.pythonfixing.com/2022/01/fixed-could-not-use-osfork-bind-several.html
##
#
import os, asyncio, socket, multiprocessing

async def handler(loop, client):
    with client:
        while True:
            data = await loop.sock_recv(client, 64)
            if not data:
                break
            await loop.sock_sendall(client, data)

# create tcp server
def create_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(('', 25000))
    sock.listen()
    sock.setblocking(False)
    return sock

# whenever accept a request ,create a handler task in eventloop
async def serving(loop, sock):
    while True:
        client, addr = await loop.sock_accept(sock)
        loop.create_task(handler(loop, client))

sock = create_server()

for num in range(multiprocessing.cpu_count() - 1):
    pid = os.fork()
    if pid <= 0:            # fork process as the same number as 
        break               # my cpu cores

loop = asyncio.get_event_loop()
loop.create_task(serving(loop, sock))
loop.run_forever()

