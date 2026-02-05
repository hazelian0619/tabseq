# 创建这个临时诊断脚本：scripts/diagnose_dataset.py
import inspect
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../src")))

try:
    from tabseq.data.tabseq_dataset import TabSeqDataset
    print("✓ 成功导入 TabSeqDataset")
    
    # 打印构造函数签名
    sig = inspect.signature(TabSeqDataset.__init__)
    print("\nTabSeqDataset.__init__ 的真实参数名：")
    print(sig)
    
    # 打印文档字符串（如果有）
    print("\n文档字符串：")
    print(inspect.getdoc(TabSeqDataset.__init__) or "无文档")
    
except ImportError as e:
    print(f"❌ 导入失败: {e}")
