* .circleci: 包含CircleCI配置文件，用于在CI/CD流程中运行测试、构建Docker镜像等操作。
* .github: 包含GitHub Action的配置文件（比如：PR自动测试等）。
* docker: 包含通过Dockerfile来创建使用预训练模型的环境的文件。
* docs: 包含文档和API说明的源代码。它们被自动转换为Transformers的官方文档。
* examples: 提供了应用Hugging Face Transformers的示例，如情感分类、序列标注、机器翻译等。
* model_cards: 包含模型说明文件，其中包括描述模型架构、输入输出格式和在哪个数据集上进行了训练等信息。
* notebooks: 包含Jupyter Notebook样例，演示了如何使用Hugging Face Transformers来完成各种任务，让用户可以方便地学习和使用这些功能。
* scripts: 包含各种辅助脚本，用于构建、安装、测试和打包Hugging Face Transformers。
* src/transformers: 是Transformers库核心代码所在的文件夹，里面包含了所有定义Transformer模型的Python代码。
* templates: 包含Hugging Face Transformers项目中使用的各种模板文件。
* tests: 包含用于测试Transformers库代码和模型的单元测试和集成测试。
* utils: 包含了一些通用工具函数，如tokenization.py（用作将文本数据预处理为模型输入）等。
