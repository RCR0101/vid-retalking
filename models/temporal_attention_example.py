"""
Example usage of the new temporal-aware cross-attention modules
"""

import torch
from transformer import (
    TemporalAwareCrossAttention, 
    TemporalTransformer, 
    EnhancedTransformer,
    EnhancedAttention
)

def example_temporal_cross_attention():
    """Example of using TemporalAwareCrossAttention directly"""
    
    # Parameters
    batch_size = 2
    temporal_len = 8
    spatial_tokens = 64  # e.g., 8x8 spatial resolution
    dim = 512
    heads = 16
    
    # Initialize temporal-aware cross-attention
    temporal_attn = TemporalAwareCrossAttention(
        dim=dim,
        heads=heads,
        dim_head=64,
        dropout=0.1,
        max_temporal_len=32
    )
    
    # Create sample data: [batch, temporal, spatial_tokens, dim]
    query_features = torch.randn(batch_size, temporal_len, spatial_tokens, dim)
    key_value_features = torch.randn(batch_size, temporal_len, spatial_tokens, dim)
    
    # Optional temporal mask (e.g., for causal attention)
    temporal_mask = torch.tril(torch.ones(batch_size, temporal_len, temporal_len))
    
    # Forward pass
    output = temporal_attn(query_features, key_value_features, temporal_mask)
    print(f"Temporal attention output shape: {output.shape}")
    
    return output


def example_temporal_transformer():
    """Example of using TemporalTransformer for video processing"""
    
    # Parameters
    batch_size = 2
    temporal_len = 16
    channels = 256
    height, width = 32, 32
    
    # Initialize temporal transformer
    temporal_transformer = TemporalTransformer(
        dim=channels,
        depth=4,
        heads=16,
        dim_head=64,
        mlp_dim=channels * 4,
        dropout=0.1,
        use_temporal_attention=True,
        max_temporal_len=32
    )
    
    # Sample video data: [batch, temporal, channels, height, width]
    query_video = torch.randn(batch_size, temporal_len, channels, height, width)
    reference_video = torch.randn(batch_size, temporal_len, channels, height, width)
    
    # Forward pass
    output = temporal_transformer(query_video, reference_video)
    print(f"Temporal transformer output shape: {output.shape}")
    
    return output


def example_enhanced_transformer():
    """Example of using EnhancedTransformer (drop-in replacement with 16+ heads)"""
    
    # Parameters
    batch_size = 4
    channels = 512
    height, width = 24, 24
    
    # Initialize enhanced transformer
    enhanced_transformer = EnhancedTransformer(
        dim=channels,
        depth=2,
        heads=16,  # Increased from default 8
        dim_head=64,
        mlp_dim=channels * 4,
        dropout=0.1
    )
    
    # Sample data: [batch, channels, height, width]
    query_features = torch.randn(batch_size, channels, height, width)
    reference_features = torch.randn(batch_size, channels, height, width)
    
    # Forward pass
    output = enhanced_transformer(query_features, reference_features)
    print(f"Enhanced transformer output shape: {output.shape}")
    
    return output


def integration_example():
    """Example of how to integrate into existing LNet architecture"""
    
    print("=== Integration Example ===")
    
    # Original LNet parameters (example)
    ngf = 64
    layer_index = 2
    original_dim = 2**(layer_index+1) * ngf  # 256
    
    print(f"Original transformer parameters:")
    print(f"  dim: {original_dim}")
    print(f"  depth: 2")
    print(f"  heads: 4")
    print(f"  dim_head: {ngf}")
    print(f"  mlp_dim: {ngf*4}")
    
    # Enhanced version
    enhanced_dim = original_dim
    enhanced_heads = 16  # Increased from 4
    enhanced_dim_head = 64  # Increased from ngf (64)
    
    print(f"\nEnhanced transformer parameters:")
    print(f"  dim: {enhanced_dim}")
    print(f"  depth: 2")
    print(f"  heads: {enhanced_heads}")
    print(f"  dim_head: {enhanced_dim_head}")
    print(f"  mlp_dim: {enhanced_dim*4}")
    
    # Create both versions for comparison
    original_params = original_dim * 2 * 4 * ngf + original_dim * 4 * ngf * 4  # Rough estimate
    enhanced_params = enhanced_dim * 2 * enhanced_heads * enhanced_dim_head + enhanced_dim * 4 * enhanced_dim * 4
    
    print(f"\nParameter comparison:")
    print(f"  Original (approx): {original_params:,}")
    print(f"  Enhanced (approx): {enhanced_params:,}")
    print(f"  Increase factor: {enhanced_params / original_params:.2f}x")


if __name__ == "__main__":
    print("Testing Temporal-Aware Cross-Attention Modules\n")
    
    # Test individual components
    print("1. Temporal Cross-Attention:")
    example_temporal_cross_attention()
    
    print("\n2. Temporal Transformer:")
    example_temporal_transformer() 
    
    print("\n3. Enhanced Transformer:")
    example_enhanced_transformer()
    
    print("\n4. Integration Analysis:")
    integration_example()
    
    print("\n✅ All examples completed successfully!")