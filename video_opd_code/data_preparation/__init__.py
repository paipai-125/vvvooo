"""
Stage 1 数据准备模块。

包含三个脚本：
- stage1_sft_template: 来源A，规则模板生成（零LLM成本）
- stage1_sft_llm_augment: 来源C，Qwen3-32B补写推理文字
- stage1_opd_filter: OPD数据双过滤pipeline
"""
