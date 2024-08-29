# TODO 跟export_jit一样的逻辑，完成flow部分的estimator的onnx导出。
# tensorrt的安装方式，再这里写一下步骤提示如下，如果没有安装，那么不要执行这个脚本，提示用户先安装，不给选择
try:
    import tensorrt
except ImportError:
    print('step1, 下载\n step2. 解压，安装whl，')
# 安装命令里tensosrt的根目录用环境变量导入，比如os.environ['tensorrt_root_dir']/bin/exetrace，然后python里subprocess里执行导出命令
# 后面我会在run.sh里写好执行命令 tensorrt_root_dir=xxxx python cosyvoice/bin/export_trt.py --model_dir xxx