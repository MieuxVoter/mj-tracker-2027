"""Test SMPData refactored"""

from mjtracker.core.smp_data import SMPData

# Test with default parameters
print("=" * 60)
print("Testing refactored SMPData...")
print("=" * 60)

try:
    smp = SMPData()
    print("\n✓ SMPData initialized successfully")
    
    print(f"\n✓ Source: {smp.source}")
    print(f"✓ Raw data shape: {smp.df_raw.shape}")
    print(f"✓ Output file: {smp.output_file}")
    
    # Test get_ranks()
    print("\n" + "=" * 60)
    print("Testing get_ranks()...")
    print("=" * 60)
    df_ranks = smp.get_ranks()
    print(f"✓ Ranks dataframe shape: {df_ranks.shape}")
    print(f"✓ Candidates: {sorted(df_ranks['candidat'].unique())}")
    print(f"✓ Date range: {df_ranks['fin_enquete'].min()} to {df_ranks['fin_enquete'].max()}")
    print("\nFirst 10 rows:")
    print(df_ranks.head(10))
    
    # Test get_intentions()
    print("\n" + "=" * 60)
    print("Testing get_intentions()...")
    print("=" * 60)
    df_intentions = smp.get_intentions()
    print(f"✓ Intentions dataframe shape: {df_intentions.shape}")
    print(f"✓ Candidates: {sorted(df_intentions['candidat'].unique())}")
    print(f"✓ Date range: {df_intentions['fin_enquete'].min()} to {df_intentions['fin_enquete'].max()}")
    print("\nFirst 10 rows:")
    print(df_intentions.head(10))
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
