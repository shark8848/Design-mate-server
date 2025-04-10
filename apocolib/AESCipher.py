import argparse
import base64
import hashlib
from Crypto import Random
from Crypto.Cipher import AES

class AESCipher:
    def __init__(self, key):
        # 设置加密块大小为32字节
        self.bs = 32
        # 使用 SHA256 哈希算法生成加密密钥
        self.key = hashlib.sha256(key.encode()).digest()

    def encrypt(self, plaintext):
        # 填充明文字符串以满足加密块大小要求
        plaintext = self._pad(plaintext)
        # 生成随机的初始化向量
        iv = Random.new().read(AES.block_size)
        # 创建 AES 对象，使用 CBC 模式和初始化向量进行加密
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        # 对填充后的明文进行加密
        ciphertext = cipher.encrypt(plaintext.encode())
        # 将初始化向量和加密后的密文以 Base64 编码形式组合起来返回
        return base64.b64encode(iv + ciphertext).decode()

    def decrypt(self, ciphertext):
        # 将 Base64 编码形式的密文解码并获取初始化向量
        ciphertext = base64.b64decode(ciphertext)
        iv = ciphertext[:AES.block_size]
        # 创建 AES 对象，使用 CBC 模式和初始化向量进行解密
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        # 对密文进行解密并去除填充，返回解密后的明文
        plaintext = cipher.decrypt(ciphertext[AES.block_size:]).decode()
        return self._unpad(plaintext)

    def _pad(self, s):
        # 计算需要填充的字节数
        padding = self.bs - len(s) % self.bs
        # 将需要填充的字节数转化成对应的填充字符
        padding_char = chr(padding) * padding
        # 返回填充后的字符串
        return s + padding_char

    def _unpad(self, s):
        # 获取字符串末尾填充字符的 ASCII 码值
        padding = ord(s[len(s) - 1:])
        # 去除填充字符并返回
        return s[:-padding]

if __name__ == '__main__':
    # 创建解析器对象
    parser = argparse.ArgumentParser(description='Encrypt or decrypt a password using AES symmetric encryption algorithm.')
    # 添加参数选项
    parser.add_argument('mode', choices=['encrypt', 'decrypt'], help='Encryption mode: encrypt or decrypt')
    parser.add_argument('--key', help='Encryption key', required=True)
    parser.add_argument('--password', help='Password to encrypt/decrypt', required=True)
    # 解析命令行参数
    args = parser.parse_args()

    # 创建 AESCipher 对象
    cipher = AESCipher(args.key)

    # 根据命令行参数进行加密或解密操作
    if args.mode == 'encrypt':
        encrypted_password = cipher.encrypt(args.password)
        print(f'Encrypted password: {encrypted_password}')
    elif args.mode == 'decrypt':
        decrypted_password = cipher.decrypt(args.password)
        print(f'Decrypted password: {decrypted_password}')
