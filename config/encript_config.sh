#!/bin/bash

# 删除旧的配置文件
rm config.yaml

# 调用配置程序生成原始配置文件 config.yaml，如果程序中断则退出脚本
python3 ../apocolib/createConfig.py || exit 1

# 调用加密程序加密生成 AESC_SERVER.yaml
python3 ../apocolib/encriptServerConfig.py config.yaml AESC_SERVER.yaml server_key.yaml

rm config.yaml
