import sys
import base64

if len(sys.argv) != 3:
    print("Usage: python encode_auth.py <username> <password>")
    sys.exit()

username = sys.argv[1]
password = sys.argv[2]
auth_str = f"{username}:{password}"
auth_bytes = auth_str.encode('ascii')
base64_bytes = base64.b64encode(auth_bytes)
auth_header = f"Basic {base64_bytes.decode('ascii')}"

print(f"Authorization header: {auth_header}")
