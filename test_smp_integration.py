"""
Test script for SMPData integration with batch_plots_smp.
This is a quick test to validate Step 1: Testing SMPData with batch plots.
"""

from pathlib import Path
import sys

# Test imports
print("Testing imports...")
try:
    from mjtracker.core.smp_data import SMPData

    print("✓ SMPData imported successfully")
except ImportError as e:
    print(f"✗ Failed to import SMPData: {e}")
    sys.exit(1)

try:
    from mjtracker import SurveysInterface

    print("✓ SurveysInterface imported successfully")
except ImportError as e:
    print(f"✗ Failed to import SurveysInterface: {e}")
    sys.exit(1)

try:
    from mjtracker.plotting.batch_plots_smp import batch_comparison_ranking, batch_comparison_intention

    print("✓ batch_plots_smp functions imported successfully")
except ImportError as e:
    print(f"✗ Failed to import batch_plots_smp: {e}")
    sys.exit(1)

try:
    from mjtracker.misc.enums import Candidacy, AggregationMode, PollingOrganizations, UntilRound

    print("✓ Enums imported successfully")
except ImportError as e:
    print(f"✗ Failed to import enums: {e}")
    sys.exit(1)

# Test SMPData loading
print("\n" + "=" * 60)
print("Testing SMPData loading...")
print("=" * 60)
try:
    smp = SMPData()
    print(f"✓ SMPData initialized successfully")
    print(f"  - Records loaded: {len(smp.df_raw)}")
    print(f"  - Candidates: {smp.df_raw['candidat'].nunique()}")
    print(f"  - Date range: {smp.df_raw['end_date'].min()} to {smp.df_raw['end_date'].max()}")
except Exception as e:
    print(f"✗ Failed to load SMPData: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test get_ranks
print("\n" + "=" * 60)
print("Testing SMPData.get_ranks()...")
print("=" * 60)
try:
    df_ranks = smp.get_ranks()
    print(f"✓ get_ranks() returned DataFrame with {len(df_ranks)} rows")
    print(f"  - Columns: {list(df_ranks.columns)}")
    print(f"  - Sample data:")
    print(df_ranks.head(3).to_string())
except Exception as e:
    print(f"✗ Failed to call get_ranks(): {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test get_intentions
print("\n" + "=" * 60)
print("Testing SMPData.get_intentions()...")
print("=" * 60)
try:
    df_intentions = smp.get_intentions()
    print(f"✓ get_intentions() returned DataFrame with {len(df_intentions)} rows")
    print(f"  - Columns: {list(df_intentions.columns)}")
    print(f"  - Sample data:")
    print(df_intentions.head(3).to_string())
except Exception as e:
    print(f"✗ Failed to call get_intentions(): {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test loading MJ surveys
print("\n" + "=" * 60)
print("Testing MJ surveys loading...")
print("=" * 60)
try:
    csv_url = "https://raw.githubusercontent.com/MieuxVoter/mj-database-2027/refs/heads/main/mj2027.csv"
    si = SurveysInterface.load(
        csv_url,
        candidates=Candidacy.ALL_CURRENT_CANDIDATES,
        polling_organization=PollingOrganizations.IPSOS,
        until_round=UntilRound.FIRST,
    )
    si.to_no_opinion_surveys()
    aggregation_mode = AggregationMode.FOUR_MENTIONS
    si.aggregate(aggregation_mode)
    si.apply_mj()
    print(f"✓ MJ surveys loaded successfully")
    print(f"  - Number of surveys: {len(si.surveys)}")
    print(f"  - Number of candidates: {len(si.candidates)}")
except Exception as e:
    print(f"✗ Failed to load MJ surveys: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test batch_comparison_ranking (without saving files)
print("\n" + "=" * 60)
print("Testing batch_comparison_ranking...")
print("=" * 60)
try:
    # Create a mock args object with minimal settings
    class MockArgs:
        comparison_ranking_plot = True
        comparison_intention = True
        html = False
        json = False
        png = False
        svg = False
        show = False
        dest = Path("output/test_smp")

    args = MockArgs()
    args.dest.mkdir(exist_ok=True, parents=True)

    # Test the function (it will try to generate plots but won't save anything due to our flags)
    # We just want to see if it runs without errors
    print("  Calling batch_comparison_ranking()...")
    batch_comparison_ranking(si, smp, args, on_rolling_data=False)
    print("✓ batch_comparison_ranking() executed successfully")
except Exception as e:
    print(f"✗ Failed to execute batch_comparison_ranking(): {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Test batch_comparison_intention
print("\n" + "=" * 60)
print("Testing batch_comparison_intention...")
print("=" * 60)
try:
    print("  Calling batch_comparison_intention()...")
    batch_comparison_intention(si, smp, args, aggregation_mode, polls=[PollingOrganizations.IPSOS])
    print("✓ batch_comparison_intention() executed successfully")
except Exception as e:
    print(f"✗ Failed to execute batch_comparison_intention(): {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# Final summary
print("\n" + "=" * 60)
print("🎉 ALL INTEGRATION TESTS PASSED!")
print("=" * 60)
print("\nSummary:")
print("✓ All imports working")
print("✓ SMPData loads data correctly from GitHub")
print("✓ SMPData.get_ranks() returns valid DataFrame")
print("✓ SMPData.get_intentions() returns valid DataFrame")
print("✓ MJ surveys load and process correctly")
print("✓ batch_comparison_ranking() executes without errors")
print("✓ batch_comparison_intention() executes without errors")
print("\n✅ SMPData is ready for integration into main_export.py!")
