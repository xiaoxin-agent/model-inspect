# model-inspect

只读模型文件检查器。首版支持旧版 **GGML v1**（文件 magic 为 `lmgg`），以本地
`ggml-model.bin` 为兼容性起点。GGUF 支持后续加入；两种格式不可混为一谈。

## 当前能力

- 解析 GGML v1 超参数和词表长度（此版本的词表项没有 score 字段）。
- 遍历 tensor 名称、维度、类型和 payload 长度；不读取或载入权重内容。
- 当前识别 F32 和 F16 tensor；遇到其他 dtype 会明确报错，不猜测长度。
- 对截断、无效维度、重复 tensor 名和越界 payload 报错。
- Python 标准库实现，无运行时第三方依赖。

## 使用

在仓库根目录运行：

```sh
python3 -m model_inspect /path/to/ggml-model.bin
python3 -m model_inspect --tensors /path/to/ggml-model.bin
python3 -m model_inspect --json /path/to/ggml-model.bin
```

也可以安装命令行入口：

```sh
python3 -m pip install .
model-inspect /path/to/ggml-model.bin
```

JSON 输出包含每个 tensor 的形状、dtype、元素数、payload 大小和文件偏移。形状按
文件顺序输出，维度 0 是最内层维度。权重文件不应提交到此仓库；大文件检查使用本地路径。

## 格式参考

GGML v1 使用 4 字节 `lmgg` magic，后跟 7 个 little-endian `uint32` 超参数。词表项是
`uint32` 长度加 token 字节；没有 score，也没有 tensor 对齐填充。Tensor 数据按 header
声明的大小跳过。格式行为参考 llama.cpp 的旧 GGML 转换器；本工具不依赖或调用该转换器。
