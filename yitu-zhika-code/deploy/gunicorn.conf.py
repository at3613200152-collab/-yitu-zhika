# gunicorn 配置（生产）
# 每个 worker 会加载一份模型（约 1.3GB）；2 worker + preload 共享父进程加载，节省内存。
bind = "127.0.0.1:8000"
workers = 2
threads = 4
timeout = 180            # 单张图 CPU 推理可能数秒，放宽超时
preload_app = True       # 启动即加载模型（fork 前），供多 worker 共享
accesslog = "-"
errorlog = "-"
loglevel = "info"
