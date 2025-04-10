import sys
from loguru import logger
import datetime

log_path = "./log/apoco.log"
#log_path = "./log/flask_server_{}.log".format(datetime.datetime.now().strftime("%Y%m%d"))
def getLogger():
    lg = logger.bind(name="apoco")
    # 清空所有设置
    lg.remove()
    # 添加控制台输出的格式,sys.stdout为输出到屏幕;关于这些配置还需要自定义请移步官网查看相关参数说明
    lg.add(sys.stdout,
                  format="<green>{time:YYYYMMDD HH:mm:ss}</green>  "  # 颜色>时间
                  #"{process.name} | "  # 进程名
                  #"{thread.name} | "  # 进程名
                               "<cyan>{module}</cyan>.<cyan>{function}</cyan>"  # 模块名.方法名
                               ":<cyan>{line}</cyan>  "  # 行号
                               "<level>{level}</level>: "  # 等级
                               "<level>{message}</level>",  # 日志内容
                  )
    # 输出到文件的格式,注释下面的add',则关闭日志写入
    lg.add(log_path, encoding='utf-8', level='DEBUG',
                        format='{time:YYYYMMDD HH:mm:ss} - '  # 时间
                               #"{process.name} | "  # 进程名
                               #"{thread.name} | "  # 进程名
                               '{module}.{function}:{line} - {level} -{message}',  # 模块名.方法名:行号
                               filter=lambda record: record["extra"]["name"] == "apoco",
                        rotation="10 MB")

    return lg

# 配置审计日志文件
audit_log_path = "./log/apoco_audit_{}.log".format(datetime.datetime.now().strftime("%Y%m%d"))
def getAuditLogger():
    lg = logger.bind(name="audit_logger")
    #lg.remove()
    # 添加输出到审计日志文件的选项
    lg.add(audit_log_path, encoding='utf-8', level='DEBUG',
           format='{time:YYYYMMDD HH:mm:ss} - '
                  '{module}.{function}:{line} - {level} -{message}',
                  filter=lambda record: record["extra"]["name"] == "audit_logger",
                  #'{level} -{message}',
           rotation="1 day")
    return lg

apoLogger = getLogger()
auditLogger = getAuditLogger()

if __name__ == '__main__':

    apoLogger.info('test log info')

    # 记录审计日志信息
    auditLogger.info('test audit log info')
