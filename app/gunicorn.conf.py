import multiprocessing
import socket
import fcntl
import struct

def get_ip_address(ifname):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    return socket.inet_ntoa(fcntl.ioctl(
        s.fileno(),
        0x8915,
        struct.pack('256s', ifname[:15], 'utf-8')
    )[20:24])

ip_adress = get_ip_address('eth0')

bind = f'{ip_adress}:5000'
workers = multiprocessing.cpu_count() * 2 + 1

timeout = 2
preload = True

loglevel = 'info'