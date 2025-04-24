## 👉🏻 有什么改动 👈🏻

1.**升级到cuda12.8适配50系显卡** 
docker目录，修改了Dockerfile和requirements适配50系显卡，升级基础镜像到cuda12.8.0，已在Ubuntu24下测试通过。
不再从git下载源码，而是在docker run时映射宿主机源码目录到/workspace/CosyVoice，方便修改和测试。

2.**接口：** 
修改了runtime下的fastapi接口，把流式和非流式接口分开了
每个接口里面新增了HttpHeader，包含模型输出的采样率
