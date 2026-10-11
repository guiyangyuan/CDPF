# Overleaf 同步记录

- 同步时间：2026-09-17 18:44 CST
- 项目地址：https://www.overleaf.com/project/6a02b4db2a09da6ad2681d51
- 主文档：`pag.tex`
- 参考文献：`cas-refs.bib`
- 论文标题：*TextConsistency: Parameter-Free Probability Fusion of Learned Prompts and Class Descriptions for Base-to-Novel Recognition*

## 同步结果

已使用 `Overleaf直接替换/` 中的同名文件覆盖项目内的 `pag.tex` 和 `cas-refs.bib`，并上传以下 3 幅论文图：

- `图1-方法框架.pdf`
- `图2-十数据集增益.pdf`
- `图3-消融稳定性效率.pdf`

文件替换后已在 Overleaf 完成重新编译。

- PDF 页数：16 页
- 编译错误：0
- 编译警告：7
- 参考文献：25 篇
- 引用状态：正文引用和参考文献已正常生成，无未定义引用

7 个警告包括 3 个 `elsarticle` 作者/通讯作者脚注命令在 PDF 书签中的 `hyperref` 字符过滤提示，以及 4 个会议文献条目未填页码的 BibTeX 提示。这些警告不影响正文、图表、引用或投稿 PDF。

## 投稿前仍需填写

论文首页仍保留作者姓名、导师姓名、学校/地址和邮箱占位符。正式投稿前应在 `pag.tex` 中替换这些信息并重新编译。

## 2026-10-10 只读核对

已在用户登录状态下重新打开上述项目并下载其当时的已编译 PDF，未覆盖或修改 Overleaf 项目。实际状态为：

- Overleaf `pag.tex` 中搜索 `ATPrompt` 为 0 结果；
- Overleaf `cas-refs.bib` 中搜索 `li2025atprompt` 为 0 结果；
- 已编译 PDF 为 16 页、25 条参考文献，包含 CuPL，但不包含 ATPrompt；
- GitHub `最终提交源码/` 为 26 条参考文献，已包含 `li2025atprompt` 和 ATPrompt/CuPL 来源说明。

因此，Overleaf 项目仍比 GitHub 最终英文源码少一次 ATPrompt 同步。本次只读核对没有擅自替换线上项目；下载的 Overleaf PDF 只代表线上当时的真实编译版本，不能冒充 GitHub 26 条文献源码的最新编译成品。

## 2026-10-11 十一数据集版本准备状态

GitHub `最终提交源码/` 已纳入当前模型的 ImageNet 三种子结果，并生成新的 Overleaf 直接替换包：

- `图2-十一数据集增益.pdf`
- `图3-十一数据集消融稳定性效率.pdf`
- 26 条参考文献，包含 `li2025atprompt`
- 十一数据集宏平均 HM：TextConsistency **78.33**，匹配 CoOp **75.06**
- ImageNet Base/Novel/HM：**76.66/71.62/74.05**

本地英文 PDF 已从该源码编译通过，但本次仍未修改 Overleaf 线上项目。在线项目完成上传和重新编译前，上述内容应视为“GitHub 待同步版本”，不应记录为已在 Overleaf 发布。
