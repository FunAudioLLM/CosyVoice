## 👉🏻 有什么改动 👈🏻

1.**升级到cuda12.8适配50系显卡** 
docker目录，修改了Dockerfile和requirements适配50系显卡，升级基础镜像到cuda12.8.0，已在Ubuntu24下测试通过。

2.**Dockerfile不从仓库下载代码**
不再从git下载源码，而是在docker run时映射宿主机源码目录到/workspace/CosyVoice，方便修改和测试。

3.**接口：** 
修改了runtime下的fastapi接口，把流式和非流式接口分开了
每个接口里面新增了HttpHeader，包含模型输出的采样率

## commit记录
本项目不定期与官方源码同步，为了保证稳定性，记录经过测试的模型和代码的commitid：

**2025/04/24**  modelscope的CosyVocie2官方模型：60b054e54afdd0d950e658dede3d2ef73d9d65b6，github代码：3bf48f125a8c25d3f9c386cdb3abf2b614391817
