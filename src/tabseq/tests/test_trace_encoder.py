"""
测试 TraceLabelEncoder 的核心功能
运行方法：
    python tests/test_trace_encoder.py
    或者（安装pytest后）：python -m pytest tests/test_trace_encoder.py -v
"""

import numpy as np
import sys
from pathlib import Path

# 让 Python 能找到 src/tabseq/...
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "src"))

from tabseq.labels.trace_encoder import TraceLabelEncoder


def test_basic_encoding():
    """测试基本编码功能"""
    encoder = TraceLabelEncoder(v_min=0.0, v_max=5.0, depth=4)
    
    # 验证基本属性
    assert encoder.n_bins == 16, f"n_bins应该是16，实际是{encoder.n_bins}"
    assert abs(encoder.bin_width - 0.3125) < 1e-6, f"bin_width应该是0.3125，实际是{encoder.bin_width}"
    
    # 测试中间值
    y = 2.5
    sequence, leaf_idx = encoder.encode(y)
    print(f"✓ encode({y}) -> sequence={sequence}, leaf_idx={leaf_idx}")
    
    # 验证sequence长度
    assert len(sequence) == 4, f"sequence长度应该是4，实际是{len(sequence)}"
    
    # 验证leaf_idx范围
    assert 0 <= leaf_idx < 16, f"leaf_idx应该在[0,15]，实际是{leaf_idx}"
    
    # 验证sequence和leaf_idx的一致性
    reconstructed_idx = 0
    for bit in sequence:
        reconstructed_idx = (reconstructed_idx << 1) | bit
    assert reconstructed_idx == leaf_idx, f"sequence重建的idx={reconstructed_idx}，但leaf_idx={leaf_idx}"
    
    print("✅ 基本编码测试通过")


def test_boundary_values():
    """测试边界值"""
    encoder = TraceLabelEncoder(v_min=0.0, v_max=5.0, depth=4)
    
    # 测试最小值
    sequence_min, leaf_idx_min = encoder.encode(0.0)
    assert leaf_idx_min == 0, f"最小值应该映射到bin 0，实际是{leaf_idx_min}"
    assert sequence_min == [0, 0, 0, 0], f"最小值的sequence应该是[0,0,0,0]，实际是{sequence_min}"
    
    # 测试最大值
    sequence_max, leaf_idx_max = encoder.encode(5.0)
    assert leaf_idx_max == 15, f"最大值应该映射到bin 15，实际是{leaf_idx_max}"
    assert sequence_max == [1, 1, 1, 1], f"最大值的sequence应该是[1,1,1,1]，实际是{sequence_max}"
    
    # 测试超出边界（应该被clip）
    _, leaf_idx_over = encoder.encode(10.0)
    assert leaf_idx_over == 15, f"超出上界应该clip到bin 15，实际是{leaf_idx_over}"
    
    _, leaf_idx_under = encoder.encode(-5.0)
    assert leaf_idx_under == 0, f"超出下界应该clip到bin 0，实际是{leaf_idx_under}"
    
    print("✅ 边界值测试通过")


def test_multi_hot_encoding():
    """测试多热编码"""
    encoder = TraceLabelEncoder(v_min=0.0, v_max=5.0, depth=4)
    
    # 测试leaf_idx=10的情况（二进制1010）
    leaf_idx = 10
    multi_hot = encoder.encode_multi_hot(leaf_idx)
    
    # 验证形状
    assert multi_hot.shape == (4, 16), f"multi_hot形状应该是(4,16)，实际是{multi_hot.shape}"
    
    # 验证每一行的和（应该是递减的：8, 4, 2, 1）
    expected_sums = [8, 4, 2, 1]
    for t in range(4):
        row_sum = multi_hot[t].sum()
        assert row_sum == expected_sums[t], f"第{t}步应该标记{expected_sums[t]}个bins，实际标记{row_sum}个"
    
    print(f"✓ leaf_idx={leaf_idx}的multi_hot每行和：{[multi_hot[t].sum() for t in range(4)]}")
    print("✅ 多热编码测试通过")


def test_encode_and_multi_hot_consistency():
    """测试encode和encode_multi_hot的一致性"""
    encoder = TraceLabelEncoder(v_min=0.0, v_max=5.0, depth=4)
    
    y = 3.14
    sequence, leaf_idx = encoder.encode(y)
    multi_hot = encoder.encode_multi_hot(leaf_idx)
    
    # 验证multi_hot最后一步只标记了一个bin（就是leaf_idx）
    last_step_marked = np.where(multi_hot[-1] == 1.0)[0]
    assert len(last_step_marked) == 1, "最后一步应该只标记1个bin"
    assert last_step_marked[0] == leaf_idx, f"最后标记的bin应该是{leaf_idx}，实际是{last_step_marked[0]}"
    
    print("✅ encode和multi_hot一致性测试通过")


def test_decode_functions():
    """测试解码功能"""
    encoder = TraceLabelEncoder(v_min=0.0, v_max=5.0, depth=4)
    
    original_y = 3.14
    sequence, leaf_idx = encoder.encode(original_y)
    decoded_y = encoder.decode_sequence(sequence)
    
    # 解码应该返回bin中心
    bin_center = encoder.decode_bin_index(leaf_idx)
    assert abs(decoded_y - bin_center) < 1e-6, "decode_sequence和decode_bin_index应该一致"
    
    # 验证解码值在原始值附近（误差不超过半个bin）
    error = abs(decoded_y - original_y)
    max_error = encoder.bin_width / 2
    assert error <= max_error, f"解码误差{error}应该≤半个bin宽度{max_error}"
    
    print(f"✓ 原始值={original_y:.4f}, 解码值={decoded_y:.4f}, 误差={error:.4f}")
    print("✅ 解码功能测试通过")


def test_degenerate_case():
    """测试退化情况（v_min = v_max）"""
    encoder = TraceLabelEncoder(v_min=3.0, v_max=3.0, depth=4)
    
    assert encoder._degenerate is True, "应该检测到退化情况"
    assert encoder.bin_width == 0.0, "退化时bin_width应该是0"
    
    sequence, leaf_idx = encoder.encode(5.0)  # 任意值
    assert sequence == [0, 0, 0, 0], "退化时sequence应该全0"
    assert leaf_idx == 0, "退化时leaf_idx应该是0"
    
    print("✅ 退化情况测试通过")


def test_different_depths():
    """测试不同的depth"""
    for depth in [1, 4, 8, 12]:
        encoder = TraceLabelEncoder(v_min=0.0, v_max=5.0, depth=depth)
        assert encoder.n_bins == 2 ** depth, f"depth={depth}时n_bins应该是{2**depth}"
        
        sequence, leaf_idx = encoder.encode(2.5)
        assert len(sequence) == depth, f"depth={depth}时sequence长度应该是{depth}"
        assert 0 <= leaf_idx < encoder.n_bins, "leaf_idx应该在有效范围内"
        
        print(f"✓ depth={depth}, n_bins={encoder.n_bins}, 测试通过")
    
    print("✅ 不同depth测试通过")


if __name__ == "__main__":
    print("=" * 100)
    print("开始测试 TraceLabelEncoder")
    print("=" * 100)
    print()
    
    try:
        test_basic_encoding()
        test_boundary_values()
        test_multi_hot_encoding()
        test_encode_and_multi_hot_consistency()
        test_decode_functions()
        test_degenerate_case()
        test_different_depths()
        
        print()
        print("=" * 100)
        print("🎉 所有测试通过！")
        print("=" * 100)
    except AssertionError as e:
        print()
        print("=" * 100)
        print(f"❌ 测试失败：{e}")
        print("=" * 100)
        import traceback
        traceback.print_exc()
