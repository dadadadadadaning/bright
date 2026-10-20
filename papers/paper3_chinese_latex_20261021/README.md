# 论文 3 中文 LaTeX 项目

本目录提供论文的 LaTeX 源码、单文件源码和已生成 PDF，可直接下载。

## 下载文件

- [完整 LaTeX 项目 ZIP](paper3_latex_source.zip?raw=true)：包含 `main.tex`、样式文件、参考文献和分章节源码，建议使用此版本。
- [论文 PDF](paper3_chinese_academic.pdf?raw=true)：已生成的 18 页中文学术排版版本。
- [单文件 LaTeX 源码](paper3_single.tex?raw=true)：适合仅粘贴或上传一个 `.tex` 文件的场景。
- [SHA-256 校验值](SHA256SUMS.txt)：用于核对以上三个文件的完整性。

## 在 Overleaf 中使用

### 完整项目

1. 下载 `paper3_latex_source.zip`。
2. 在 Overleaf 中选择“新建项目”→“上传项目”，上传该 ZIP。
3. 将主文件设为 `main.tex`，编译器设为 **XeLaTeX**，然后重新编译。

### 单文件版本

1. 下载 `paper3_single.tex`，在 Overleaf 新建空白项目。
2. 将文件重命名为 `main.tex` 后上传，或将其全部内容粘贴到项目的 `main.tex`。
3. 将编译器设为 **XeLaTeX**，然后重新编译。

## 内容与版本说明

- 当前 PDF 共 **18 页**，包含 **14 个公式、8 张表、50 条参考文献**。
- 源码中保留 **10 个图位**，尚未嵌入实际图片；图位是占位内容。
- 原稿中的“待测”内容按原样保留，尚未替换为实验实测结果。
- 本版内容来自 **《论文3_按当前代码重写.docx》**，未更新聊天中后续提供的实验结果。
- 本目录仅包含论文交付文件、使用说明和校验值。

## 校验文件

在下载目录执行以下命令可核对三个交付文件：

```bash
sha256sum -c SHA256SUMS.txt
```
