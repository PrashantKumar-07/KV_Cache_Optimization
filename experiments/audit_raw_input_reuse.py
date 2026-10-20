"""Public whole-decode allocation/root audit of the isolated payload prototype."""
import audit_exact_pool_resources as resources
import compare_raw_input_reuse as comparison


def main():
    previous = resources.comparison.main
    resources.comparison.main = comparison.main
    try: resources.main()
    finally: resources.comparison.main = previous


if __name__=='__main__': main()
