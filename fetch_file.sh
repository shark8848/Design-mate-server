#git fetch origin <branch_name>
#git fetch origin <branch_name>
#git checkout FETCH_HEAD -- <file_path>#git checkout FETCH_HEAD -- <file_path>

#!/bin/bash
# 主要用于将 generalMessageService 部署在 apococloud01 上，自动下载程序
# 定义变量
BRANCH_NAME='develop-sunhy'
FILE_LIST='file_list.txt'
GITLAB_USERNAME='sunhy'
GITLAB_PASSWORD='sunhy@2023'

# 从远程仓库获取最新代码，需要身份验证
echo "Authenticating with GitLab..."
git config credential.helper store # 记录密码以避免多次输入
git fetch --all # 获取所有远程分支和标签
git checkout $BRANCH_NAME # 切换到指定分支
echo "Authenticated successfully."

# 循环遍历清单中的每个文件名
while read FILE_NAME
do
  # 检出指定文件
  git checkout FETCH_HEAD -- "$FILE_NAME"
done < $FILE_LIST

# 恢复 Git 的默认凭据缓存设置
git config --unset credential.helper
